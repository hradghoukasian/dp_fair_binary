# dp_fair_binary

This repository contains the code implementation of the paper "Differentially Private Fair Binary Classifications".

The full version of the paper is available on [arXiv](https://arxiv.org/abs/2402.15603).

## Algorithm 2

To run the code corresponding to Algorithm 2 in the paper, you need to set proper parameters and hyperparameters. First, configure the DP-SGD parameters, including the clipping constant, standard deviation of the Gaussian noise, learning rate as well as ε₀, and ε₁ in the Algorithm2.py file. Additionally, specify the dataset, number of epochs, and number of trained models in the args.py file. After setting the parameters, run the model using the following command:

```bash
python3 Algorithm2.py
```

## Privacy Accounting Methods

To determine the proper standard deviation of the Gaussian noise used in DP-SGD based on the PRV Accountant method, refer to PRVAccountant.ipynb. Then, set the noise value accordingly. To switch to the moments accountant privacy accounting method, modify the initialize_accountant function in Algorithm2.py to return RDPAccountant(). For the GDP Accountant, set it to return GaussianAccountant().

## DP-FERMI Method

To run the code corresponding to the DP-FERMI method, configure all hyperparameters including the dataset, learning rates, privacy parameters (ε and δ), and the range of fairness to accuracy tradeoff (λ \lambda). After setting the parameters, execute the model using the following command:

```bash
python3 dp_fermi.py
```

This README provides comprehensive instructions for running the code in this repository. If you encounter any issues or have questions, feel free to reach out via email at [ghoukash@mcmaster.ca](mailto:ghoukash@mcmaster.ca) or refer to the paper for further details.

