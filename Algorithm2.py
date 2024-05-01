import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant
from opacus.accountants import RDPAccountant
from sklearn.metrics import accuracy_score
from dataloader import GeneralData
from args import Args
import pandas as pd
from torch.distributions import Laplace


class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


args = Args()
args.assign()
args.delta = 1e-5
args.epsilon_0= 0.05
args.epsilon_1 = 0.05
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_std = 3.13
clipping_constant = 1.5
full_data = GeneralData(path=args.path, sensitive_attributes=args.sensitive_attributes,
                        cols_to_norm=args.cols_to_norm, output_col_name=args.output_col_name, split=args.split)

train_data = full_data.getTrain()
test_data = full_data.getTest()

train_size = int((2 / 3) * len(train_data))
post_process_size = len(train_data) - train_size
dataset_train, dataset_post_process = random_split(train_data, [train_size, post_process_size])
dataset_test = test_data

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_post_process = DataLoader(dataset_post_process, batch_size=args.batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

# Calculate the total number of batches per epoch
# Note: The DataLoader automatically handles the last batch that may be smaller than args.batch_size
total_batches_per_epoch = len(dataloader_train)

# Calculate the total number of optimization steps
num_optimization_steps = total_batches_per_epoch * args.epochs

# Compute the probability of a data point being in a batch
batch_probability = args.batch_size / len(dataset_train)

# Print the probability
print(f"Probability of a data point being in a batch: {batch_probability:.4f}")

# Print the number of optimization steps
print(f"Total number of optimization steps: {num_optimization_steps}")


num_non_sensitive_features = len(full_data.non_sens_attr)
num_sensitive_features = len(full_data.sensitive_attributes)
num_total_features = num_non_sensitive_features + num_sensitive_features

# Moments Accountant
def initialize_accountant():
    return RDPAccountant()

# Gaussian Accountant
# def initialize_accountant():
#     return GaussianAccountant()


alphas = []
betas = []
accuracies = []
demographic_parities = []

for model_run in range(args.num_models_train):
    print(f"Training model {model_run + 1}/{args.num_models_train}")

    model = LogisticRegression(num_total_features).to(args.device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine()

    accountant = initialize_accountant()
    model, optimizer, dataloader_train = privacy_engine.make_private(module=model, optimizer=optimizer,
                                                                     data_loader=dataloader_train, noise_multiplier=noise_std/clipping_constant,
                                                                     max_grad_norm=clipping_constant)

    model.train()
    for epoch in range(args.epochs):
        for batch in dataloader_train:
            features, sensitive_attributes, labels, _ = batch
            sensitive_attribute = sensitive_attributes[:, 1]
            features, sensitive_attribute, labels = features.to(args.device), sensitive_attribute.to(
                args.device), labels.to(args.device)
            combined_input = torch.cat((features.float(), sensitive_attribute.unsqueeze(1).float()), dim=1)
            outputs = model(combined_input).squeeze()
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accountant.step(noise_multiplier=noise_std / clipping_constant, sample_rate=args.batch_size / len(dataset_train))

    total_sensitive_0_post = 0
    total_sensitive_1_post = 0
    label_1_given_sensitive_0_post = 0
    label_1_given_sensitive_1_post = 0


    model.eval()
    with torch.no_grad():
        for features, sensitive_attributes, labels, _ in dataloader_post_process:
            sensitive_attribute = sensitive_attributes[:, 1]
            features, sensitive_attribute, labels = features.to(args.device), sensitive_attribute.to(
                args.device), labels.to(args.device)
            combined_input = torch.cat((features.float(), sensitive_attribute.unsqueeze(1).float()), dim=1)
            outputs = model(combined_input).squeeze()
            predicted_labels = (outputs > 0.5).float()

            total_sensitive_0_post += torch.sum(sensitive_attribute == 0).item()
            total_sensitive_1_post += torch.sum(sensitive_attribute == 1).item()
            label_1_given_sensitive_0_post += torch.sum((sensitive_attribute == 0) & (predicted_labels == 1)).item()
            label_1_given_sensitive_1_post += torch.sum((sensitive_attribute == 1) & (predicted_labels == 1)).item()



    alpha = label_1_given_sensitive_0_post / total_sensitive_0_post if total_sensitive_0_post > 0 else 0
    beta = label_1_given_sensitive_1_post / total_sensitive_1_post if total_sensitive_1_post > 0 else 0

    # Add Laplace noise to alpha and beta
    laplace_dist_alpha = Laplace(torch.tensor([0.0]), torch.tensor([1 / (total_sensitive_0_post * args.epsilon_0)]))
    laplace_dist_beta = Laplace(torch.tensor([0.0]), torch.tensor([1 / (total_sensitive_1_post * args.epsilon_1)]))
    alpha_noisy = alpha + laplace_dist_alpha.sample().item()
    beta_noisy = beta + laplace_dist_beta.sample().item()
    alpha_noisy = max(0, min(1, alpha_noisy))
    beta_noisy = max(0, min(1, beta_noisy))


    predictions, labels_list = [], []
    sensitive_0_predicted_1 = 0
    sensitive_1_predicted_1 = 0
    sensitive_0_total = 0
    sensitive_1_total = 0

    with torch.no_grad():
        for features, sensitive_attributes, labels, _ in dataloader_test:
            sensitive_attribute = sensitive_attributes[:, 1]
            features, sensitive_attribute, labels = features.to(args.device), sensitive_attribute.to(
                args.device), labels.to(args.device)
            combined_input = torch.cat((features.float(), sensitive_attribute.unsqueeze(1).float()), dim=1)
            outputs = model(combined_input).squeeze()
            original_predicted = (outputs > 0.5).float()
            modified_predicted = original_predicted.clone()


            if alpha_noisy >= beta_noisy:
                for i in range(len(original_predicted)):
                    s = torch.rand(1).item()
                    if sensitive_attribute[i] == 0 and original_predicted[i] == 1 and (alpha_noisy != 0) and s > (
                            (alpha_noisy + beta_noisy) / (2 * alpha_noisy)) :
                        modified_predicted[i] = 0
                    elif sensitive_attribute[i] == 1 and original_predicted[i] == 0 and (beta_noisy != 1) and s <= (
                            (alpha_noisy - beta_noisy) / (2 * (1 - beta_noisy))) :
                        modified_predicted[i] = 1

            else:
                for i in range(len(original_predicted)):
                    s = torch.rand(1).item()
                    if sensitive_attribute[i] == 1 and original_predicted[i] == 1 and (beta_noisy != 0) and s > (
                            (alpha_noisy + beta_noisy) / (2 * beta_noisy)):
                        modified_predicted[i] = 0
                    elif sensitive_attribute[i] == 0 and original_predicted[i] == 0 and (alpha_noisy != 1) and s <= (
                            (beta_noisy - alpha_noisy ) / (2 * (1 - alpha_noisy))):
                        modified_predicted[i] = 1





            sensitive_0_predicted_1 += torch.sum((sensitive_attribute == 0) & (modified_predicted == 1)).item()
            sensitive_1_predicted_1 += torch.sum((sensitive_attribute == 1) & (modified_predicted == 1)).item()
            sensitive_0_total += torch.sum(sensitive_attribute == 0).item()
            sensitive_1_total += torch.sum(sensitive_attribute == 1).item()
            predictions.extend(modified_predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, predictions)
    pr_sensitive_0 = sensitive_0_predicted_1 / sensitive_0_total if sensitive_0_total > 0 else 0
    pr_sensitive_1 = sensitive_1_predicted_1 / sensitive_1_total if sensitive_1_total > 0 else 0
    demographic_parity_violation = abs(pr_sensitive_0 - pr_sensitive_1)
    epsilon = accountant.get_epsilon(delta=args.delta)

    print(f"Model {model_run + 1}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Demographic Parity Violation: {demographic_parity_violation:.4f}")
    print(f"Model Privacy (ε = {epsilon + args.epsilon_0 + args.epsilon_1 :.2f}, δ = {args.delta})")
    print("--------------------------------------------")

    alphas.append(alpha_noisy)
    betas.append(beta_noisy)
    accuracies.append(accuracy)
    demographic_parities.append(demographic_parity_violation)

# results_df = pd.DataFrame({
#     'Alpha': alphas,
#     'Beta': betas,
#     'Accuracy': accuracies,
#     'Demographic Parity Violation': demographic_parities
# })
#
# print(results_df)

# Added code to calculate and print averages
average_accuracy = sum(accuracies) / len(accuracies)
average_demographic_parity_violation = sum(demographic_parities) / len(demographic_parities)

print(f"Average Accuracy: {average_accuracy:.4f}")
print(f"Average Demographic Parity Violation: {average_demographic_parity_violation:.4f}")
