import numpy as np
from dataloader import *


class Args():
    def __init__(self):
        self.debug = True
        self.tuning = False
        self.num_models_train = 10
        self.demographic_parity = True  # Put False for the Equalized Odds experiments
        self.epochs = 50
        self.batch_size = 1024
        self.split = 0.75
        self.val_interval = 10
        self.num_layers = 1

        # No tuning required, since they are tradeoff parameters which is what we need to assess for all datasets
        self.epsilon = None
        #self.epsilon_list = [0.5, 1, 3, 9, np.inf]
        self.epsilon_list = [3]
        ########################################################################################

        # Calculated Parameters, hence require no tuning
        self.delta = None
        self.std_theta = None
        self.std_W = None
        ###############################################

        # Parameters which require tuning (prefrably for each value of the pair {lambda, epsilon})
        self.C = 5
        self.lipschitz_theta = 5

        self.lr_theta = 0.005  # 0.001 for large scale
        self.lr_theta_list = None

        self.lr_W = 0.01  # 0.005 for large scale
        self.lr_W_list = None

        self.lambd = None
        self.lambd_list = [0.5]
        #########################################################################################

        self.dataset = "adult"  # One of ["adult", "credit-card"]

    def assign(self):
        if self.num_layers:
            if self.num_layers == 1:
                self.model_type = "logistic-regression"

        if not self.lambd_list:
            self.lambd_list = [self.lambd]
        if not self.epsilon_list:
            self.epsilon_list = [self.epsilon]

        if not self.lr_W_list:
            self.lr_W_list = [self.lr_W]
        if not self.lr_theta_list:
            self.lr_theta_list = [self.lr_theta]

        if self.dataset == "adult":
            self.num_inp_attr = 102
        if self.dataset == "credit-card":
            self.num_inp_attr = 85
        
        self.out_attr = 1

        if self.dataset == "adult":
            self.path = "./Datasets/Adult/adult_original_purified.csv"
        if self.dataset == "credit-card":
            self.path = "./Datasets/CreditCard/credit-card-defaulters_processed.csv"
        

        if self.dataset == "adult":
            self.cols_to_norm = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        if self.dataset == "credit-card":
            self.cols_to_norm = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
                                 "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
      
        if self.dataset == "adult":
            self.sensitive_attributes = ["sex"]
        if self.dataset == "credit-card":
            self.sensitive_attributes = ["SEX"]
    

        if self.dataset == "adult":
            self.output_col_name = ">50K"
        if self.dataset == "credit-card":
            self.output_col_name = "default payment next month"
        

    def calculate_noise(self):
        if self.dataset in ["adult","credit-card"]:
            full_data = GeneralData(path=self.path, sensitive_attributes=self.sensitive_attributes,
                                    cols_to_norm=self.cols_to_norm, output_col_name=self.output_col_name,
                                    split=self.split)
            dataset_train = full_data.getTrain()
        
        dataloader_train = Data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        n = dataset_train.__len__()
        T = self.epochs * (int(n / self.batch_size) + (1 if n % self.batch_size != 0 else 0))
        sensitive_index_all = []
        for non_sensitive, sensitive, label, sensitive_index in dataloader_train:
            sensitive_index_all.extend(sensitive_index.squeeze().tolist())

        k = np.max(sensitive_index_all) + 1
        l = self.out_attr

        count_dict = {}
        for attr in sensitive_index_all:
            try:
                count_dict[attr] += 1
            except KeyError:
                count_dict[attr] = 1

        min_count = n + 1
        for _, coun in count_dict.items():
            min_count = min(min_count, coun)

        rho = (min_count - 1) / n

        self.delta = 1e-5

        # set std_theta and std_W to guarantee privacy w.r.t. all features
        self.std_W = ((32 * T * np.log(1 / self.delta)) * (self.C ** 2 + (1 / rho)) / (
                    self.epsilon ** 2 * n ** 2)) ** 0.5

        self.std_theta = (((64 * self.lipschitz_theta ** 2 * self.C ** 2 * np.log(1 / self.delta) * T) / (
                    self.epsilon ** 2 * n ** 2 * rho))
                          + ((32 * self.C ** 4 * self.lipschitz_theta ** 2 * 4 * np.log(1 / self.delta) * T) / (
                            self.epsilon ** 2 * n ** 2))) ** 0.5

        # set std_theta and std_W to guarantee privacy w.r.t. sensitive attribute
        # self.std_theta = ((16 * self.lipschitz_theta**2 * self.C**2 * np.log(1/self.delta) * T)/(self.epsilon**2 * n**2 * rho))**0.5
        # self.std_W = ((16 * T * np.log(1/self.delta))/(self.epsilon**2 * n**2 * rho))**0.5