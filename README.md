# Differentially Private Fair Binary Classifications

This repository contains the code implementation for the paper *Differentially Private Fair Binary Classifications*, published in **IEEE ISIT 2024** ([IEEE](https://ieeexplore.ieee.org/abstract/document/10619147)).  
The full version of the paper is available on [arXiv](https://arxiv.org/abs/2402.15603).

## 📖 Overview

We study binary classification under the joint constraints of **differential privacy** and **fairness**.  
We first design a fairness-only algorithm using the decoupling technique, which aggregates classifiers trained on different demographic groups into a single classifier satisfying statistical parity. We then refine this method to incorporate differential privacy while maintaining strong fairness guarantees.  

Our proposed **Algorithm 2** achieves **competitive accuracy** compared to the state-of-the-art DP-Fair classification method **DP-FERMI**, while **significantly improving fairness** for the same privacy and accuracy levels.

---

## 📊 Experimental Results

Through several experiments on two well-known datasets (**Adult** and **Credit Card**), we demonstrate that **Algorithm 2** achieves competitive accuracy compared to **DP-FERMI**.  
In particular, for a given level of accuracy and privacy, our algorithm provides a significantly better fairness guarantee across both datasets.

<table>
<tr>
<td>

<p align="center"><b>Adult Dataset (ε′ = 3, δ′ = 10⁻⁵)</b></p>

| Method              | Accuracy | Statistical Parity Gap |
|---------------------|:--------:|:----------------------:|
| Algorithm 2         | 0.7752   | 0.0057                 |
| DP-FERMI (λ = 0.5)  | 0.7998   | 0.1020                 |
| DP-FERMI (λ = 1)    | 0.7859   | 0.0462                 |
| DP-FERMI (λ = 1.5)  | 0.7822   | 0.0267                 |
| DP-FERMI (λ = 1.9)  | 0.7749   | 0.0126                 |
| DP-FERMI (λ = 2.5)  | 0.7673   | 0.0099                 |

</td>
<td>

<p align="center"><b>Adult Dataset (ε′ = 9, δ′ = 10⁻⁵)</b></p>

| Method              | Accuracy | Statistical Parity Gap |
|---------------------|:--------:|:----------------------:|
| Algorithm 2         | 0.7782   | 0.0054                 |
| DP-FERMI (λ = 0.5)  | 0.8091   | 0.0944                 |
| DP-FERMI (λ = 1)    | 0.7923   | 0.0413                 |
| DP-FERMI (λ = 1.5)  | 0.7810   | 0.0152                 |
| DP-FERMI (λ = 1.7)  | 0.7782   | 0.0121                 |
| DP-FERMI (λ = 2.5)  | 0.7693   | 0.0030                 |

</td>
</tr>
</table>

<table>
<tr>
<td>

<p align="center"><b>Credit Card Dataset (ε′ = 3, δ′ = 10⁻⁵)</b></p>

| Method              | Accuracy | Statistical Parity Gap |
|---------------------|:--------:|:----------------------:|
| Algorithm 2         | 0.7842   | 0.0041                 |
| DP-FERMI (λ = 0.1)  | 0.7899   | 0.0212                 |
| DP-FERMI (λ = 0.2)  | 0.7846   | 0.0193                 |
| DP-FERMI (λ = 0.5)  | 0.7777   | 0.0185                 |
| DP-FERMI (λ = 1)    | 0.7759   | 0.0105                 |
| DP-FERMI (λ = 2.5)  | 0.7669   | 0.0110                 |

</td>
<td>

<p align="center"><b>Credit Card Dataset (ε′ = 9, δ′ = 10⁻⁵)</b></p>

| Method              | Accuracy | Statistical Parity Gap |
|---------------------|:--------:|:----------------------:|
| Algorithm 2         | 0.7908   | 0.0071                 |
| DP-FERMI (λ = 0.25) | 0.7996   | 0.0188                 |
| DP-FERMI (λ = 0.35) | 0.7950   | 0.0174                 |
| DP-FERMI (λ = 0.5)  | 0.7912   | 0.0172                 |
| DP-FERMI (λ = 1)    | 0.7895   | 0.0105                 |
| DP-FERMI (λ = 2.5)  | 0.7884   | 0.0066                 |

</td>
</tr>
</table>

---

## ⚙️ Running the Code

### Algorithm 2: Private and Fair Binary Classifier with Utility Gap Guarantee

To run **Algorithm 2**:

1. Configure DP-SGD parameters in `Algorithm2.py`:  
   - Clipping constant  
   - Standard deviation of Gaussian noise  
   - Learning rate  
   - ε₀ and ε₁  

2. In `args.py`, specify:
   - Dataset  
   - Number of epochs  
   - Number of trained models  

3. Then run:
```bash
python3 Algorithm2.py
```
---

**Note:** Before running Algorithm 2, you must determine the appropriate noise level for DP-SGD using one of the following privacy accounting methods:

- **PRV Accountant**  
  Run `PRVAccountant.ipynb` to compute the appropriate noise level for a given (ε, δ).

- **Moments Accountant**  
  In `Algorithm2.py`, modify `initialize_accountant()` to return `RDPAccountant()`.

- **GDP Accountant**  
  In `Algorithm2.py`, set `initialize_accountant()` to return `GaussianAccountant()`.


  ---

  ## 🧮 DP-FERMI Method

To run the **DP-FERMI** baseline:

1. Set all hyperparameters in `dp_fermi.py`:
   - Dataset  
   - Learning rates  
   - Privacy parameters (ε, δ)  
   - Fairness–accuracy trade-off parameter λ  

2. Run:
```bash
python3 dp_fermi.py
```

---
📄 Citation
If you use this code or find our work helpful, please cite:

bibtex
Copy
Edit
@inproceedings{ghoukasian2024differentially,
  title     = {Differentially Private Fair Binary Classifications},
  author    = {Ghoukasian, Hrad and Asoodeh, Shahab},
  booktitle = {2024 IEEE International Symposium on Information Theory (ISIT)},
  pages     = {611--616},
  year      = {2024},
  organization = {IEEE}
}
