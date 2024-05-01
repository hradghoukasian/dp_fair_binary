import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from pathlib import Path

from args import Args
from models import *
from dataloader import *
from metrics import *

args = Args()
args.assign()

if args.dataset in ["adult", "credit-card"]:
    full_data = GeneralData(path=args.path, sensitive_attributes=args.sensitive_attributes,
                            cols_to_norm=args.cols_to_norm, output_col_name=args.output_col_name, split=args.split)
    dataset_train = full_data.getTrain()
    dataset_test = full_data.getTest()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accuracy_list = []
demographic_parity_list = []


for args.lr_theta in args.lr_theta_list:
    for args.lr_W in args.lr_W_list:
        for args.epsilon in args.epsilon_list:
            # Calculating noise based on the changed values of epsilon
            data_table = {}
            args.calculate_noise()
            print(
                f'''Learning Rate Theta: {args.lr_theta} Learning Rate W: {args.lr_W} Epsilon: {args.epsilon} \n\tStd Theta: {args.std_theta} Std W: {args.std_W}''')

            for args.lambd in args.lambd_list:

                data_table[args.lambd] = {}
                data_table[args.lambd]["demographic_parity_list"] = []
                data_table[args.lambd]["equalized_odds_list"] = []
                data_table[args.lambd]["equalized_opportunity_list"] = []
                data_table[args.lambd]["misclassification_error_list"] = []

                for model_number in range(args.num_models_train):
                    torch.manual_seed(model_number)
                    # Setting up new initializations and shuffling
                    if args.tuning:
                        dataloader_train = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=2)
                        dataloader_test = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=2)
                    else:
                        dataloader_train = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=2)
                        dataloader_test = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=2)

                    # Model, Loss Critetions, Optimizers, and Parameters
                    if args.model_type == "logistic-regression":
                        model = LogisticRegression(args.num_inp_attr)

                    model = model.to(device)

                    if args.model_type == "logistic-regression":
                        classification_loss_fn = nn.BCEWithLogitsLoss().to(device)
                    else:
                        classification_loss_fn = nn.CrossEntropyLoss().to(device)
                    W = nn.Parameter(data=torch.randn(args.out_attr, args.out_attr).to(device), requires_grad=True)
                    if not args.demographic_parity:
                        W_ = nn.Parameter(data=torch.randn(args.out_attr, args.out_attr).to(device), requires_grad=True)

                    if args.model_type == "logistic-regression":
                        P_s_negative_half = full_data.calculateP_s(args.demographic_parity)
                    else:
                        P_s_negative_half = dataset_train.calculateP_s(args.demographic_parity)

                    if args.demographic_parity:
                        P_s_negative_half = P_s_negative_half.to(device)
                    else:
                        P_s_negative_half[0] = P_s_negative_half[0].to(device)
                        P_s_negative_half[1] = P_s_negative_half[1].to(device)

                    for epoch in tqdm(range(args.epochs), f"Lambd: {args.lambd} Model {model_number}"):
                        # Training Loop
                        for batch_no, (non_sensitive, sensitive, label, _) in enumerate(dataloader_train):
                            non_sensitive = non_sensitive.to(device)
                            sensitive = sensitive.to(device)
                            label = label.to(device)

                            model.zero_grad()
                            if W.grad is not None:
                                W.grad.copy_(torch.zeros(W.shape))
                            if not args.demographic_parity:
                                if W_.grad is not None:
                                    W_.grad.copy_(torch.zeros(W_.shape))

                            y_logit, y_hat = model(non_sensitive.float())

                            if args.model_type == "logistic-regression":
                                classification_loss = classification_loss_fn(y_logit, label.unsqueeze(1).float())
                            else:
                                classification_loss = classification_loss_fn(y_logit, label)

                            if args.demographic_parity:
                                p_hat_yhat = torch.diag(torch.mean(y_hat, axis=0))
                                p_hat_yhat_s = 1 / y_hat.size(0) * y_hat.T @ sensitive
                                fermi_loss = -1 * torch.trace(W @ p_hat_yhat @ W.T) + 2 * torch.trace(
                                    W @ p_hat_yhat_s @ P_s_negative_half) - 1
                            else:
                                # We cover equalized odds only for the binary classification case, this can be easily extended to a multiclass setting
                                y_hat_given_1 = []
                                sensitive_given_1 = []
                                y_hat_given_0 = []
                                sensitive_given_0 = []
                                for i in range(label.size(0)):
                                    if label[i] == 1:
                                        y_hat_given_1.append(y_hat[i].unsqueeze(0))
                                        sensitive_given_1.append(sensitive[i].unsqueeze(0))
                                    else:
                                        y_hat_given_0.append(y_hat[i].unsqueeze(0))
                                        sensitive_given_0.append(sensitive[i].unsqueeze(0))
                                y_hat_given_1 = torch.cat(y_hat_given_1, axis=0)
                                y_hat_given_0 = torch.cat(y_hat_given_0, axis=0)
                                sensitive_given_1 = torch.cat(sensitive_given_1, axis=0)
                                sensitive_given_0 = torch.cat(sensitive_given_0, axis=0)
                                p_hat_yhat_part_1 = torch.diag(torch.mean(y_hat_given_1, axis=0))
                                p_hat_yhat_part_0 = torch.diag(torch.mean(y_hat_given_0, axis=0))
                                p_hat_yhat_s_given_1 = 1 / y_hat_given_1.size(0) * y_hat_given_1.T @ sensitive_given_1
                                p_hat_yhat_s_given_0 = 1 / y_hat_given_0.size(0) * y_hat_given_0.T @ sensitive_given_0
                                fermi_loss_1 = -1 * torch.trace(W @ p_hat_yhat_part_1 @ W.T) + 2 * torch.trace(
                                    W @ p_hat_yhat_s_given_1 @ P_s_negative_half[1]) - 1
                                fermi_loss_0 = -1 * torch.trace(W_ @ p_hat_yhat_part_0 @ W_.T) + 2 * torch.trace(
                                    W_ @ p_hat_yhat_s_given_0 @ P_s_negative_half[0]) - 1
                                fermi_loss = fermi_loss_0 + fermi_loss_1

                            total_loss = classification_loss + args.lambd * fermi_loss
                            total_loss.backward()

                            with torch.no_grad():
                                grad_norm = 0
                                for param in model.parameters():
                                    grad_norm += torch.norm(param.grad) ** 2
                                grad_norm = grad_norm ** 0.5

                                # To satisfy the Lipschitzness of the Loss with respect to theta
                                divide_by = grad_norm.item() / args.lipschitz_theta
                                divide_by = divide_by if divide_by > 1 else 1

                                for param in model.parameters():
                                    if args.std_theta != 0:
                                        u_t = torch.normal(mean=0, std=args.std_theta, size=param.shape).to(device)
                                    else:
                                        u_t = torch.zeros_like(param).to(device)
                                    param.sub_(args.lr_theta * (param.grad / divide_by + u_t))

                                if args.std_W != 0:
                                    v_t = torch.normal(mean=0, std=args.std_W, size=W.shape).to(device)
                                else:
                                    v_t = torch.zeros_like(W).to(device)
                                W.add_(args.lr_W * (W.grad + args.lambd * v_t))

                                if not args.demographic_parity:
                                    if args.std_W != 0:
                                        v_t = torch.normal(mean=0, std=args.std_W, size=W_.shape).to(device)
                                    else:
                                        v_t = torch.zeros_like(W_).to(device)
                                    W_.add_(args.lr_W * (W_.grad + args.lambd * v_t))

                                # Projecting W into a convex space
                                norm_W = torch.norm(W.data)
                                if norm_W > args.C:
                                    W.copy_(args.C * W.data / norm_W)

                                if not args.demographic_parity:
                                    norm_W_ = torch.norm(W_.data)
                                    if norm_W_ > args.C:
                                        W_.copy_(args.C * W_.data / norm_W_)

                    model.eval()

                    # Evaluation for data table
                    sensitive_index_all = []
                    y_hat_all = []
                    label_all = []
                    for non_sensitive, sensitive, label, sensitive_index in dataloader_test:
                        non_sensitive = non_sensitive.to(device)
                        sensitive = sensitive.to(device)
                        label = label.to(device)
                        sensitive_index = sensitive_index.to(device)

                        with torch.no_grad():
                            y_logit, y_hat = model(non_sensitive.float())

                        sensitive_index_all.extend(sensitive_index.squeeze().tolist())
                        if args.model_type == "logistic-regression":
                            y_hat_all.extend((y_hat.detach().cpu() > 0.5).squeeze().tolist())
                        else:
                            y_hat_all.extend(y_hat.detach().cpu().squeeze().tolist())
                        label_all.extend(label.squeeze().tolist())
                    if args.model_type == "logistic-regression":
                        y_hat_all = [1 if u else 0 for u in y_hat_all]
                        demographic_parity = demographic_parity_violation_binary(sensitive_index_all, y_hat_all,
                                                                                 label_all)
                        equalized_odds = equalized_odds_violation_binary(sensitive_index_all, y_hat_all, label_all)
                        # equalized_opportunity = equalized_opportunity_violation_binary(sensitive_index_all, y_hat_all, label_all)
                        misclassification_error = 1 - accuracy(y_hat_all, label_all)
                    else:
                        demographic_parity = demographic_parity_violation_multiple(sensitive_index_all, y_hat_all,
                                                                                   label_all)
                        equalized_odds = equalized_odds_violation_multiple(sensitive_index_all, y_hat_all, label_all)
                        # equalized_opportunity = equalized_opportunity_violation_multiple(sensitive_index_all, y_hat_all, label_all)
                        misclassification_error = 1 - accuracy(y_hat_all, label_all)

                    print(f"Demographic Parity: {demographic_parity}")
                    # print(f"Equalized Odds: {equalized_odds}")
                    # print(f"Equalized Opportunity: {equalized_opportunity}")
                    print(f"Misclassification Error: {misclassification_error}")

                    accuracy_list.append(1 - misclassification_error)
                    demographic_parity_list.append(demographic_parity)

                    model.train()

                    data_table[args.lambd]["demographic_parity_list"].append(demographic_parity)
                    data_table[args.lambd]["equalized_odds_list"].append(equalized_odds)
                    # data_table[args.lambd]["equalized_opportunity_list"].append(equalized_opportunity)
                    data_table[args.lambd]["misclassification_error_list"].append(misclassification_error)

                data_table[args.lambd]["demographic_parity_list"] = torch.tensor(
                    data_table[args.lambd]["demographic_parity_list"])
                data_table[args.lambd]["equalized_odds_list"] = torch.tensor(
                    data_table[args.lambd]["equalized_odds_list"])
                # data_table[args.lambd]["equalized_opportunity_list"] = torch.tensor(data_table[args.lambd]["equalized_opportunity_list"])
                data_table[args.lambd]["misclassification_error_list"] = torch.tensor(
                    data_table[args.lambd]["misclassification_error_list"])

                _ = Path(f"./PlotConstructionData/DP-FERMI/{args.dataset}").mkdir(parents=True, exist_ok=True)
                torch.save(data_table,
                           f"./PlotConstructionData/DP-FERMI/{args.dataset}/eps_{args.epsilon}_{args.model_type}.pt")



# Compute the average accuracy and average demographic parity
average_accuracy = sum(accuracy_list) / len(accuracy_list)
average_demographic_parity = sum(demographic_parity_list) / len(demographic_parity_list)

# Print the results
print("--------------")
print(f"Average Accuracy: {average_accuracy}")
print(f"Average Demographic Parity: {average_demographic_parity}")