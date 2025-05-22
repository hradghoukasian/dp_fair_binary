
# Reproducibility Instructions

This repository contains code to reproduce all experiments and figures in the paper.

## Overview

- `exp_*.py` scripts generate experiment data.
- `plot_*.py` scripts generate plots from the saved data.

## Figure: Laplace Pure vs Functional (Local vs Global)

To generate the output distributions and visualize Figure **`Laplace pure and functional local vs global`**:

```bash
python exp_LapMixture_visual.py
python plot_LapMixture_visual.py
```

## Finite Space Results

Six figures are generated for various values of \( k \) under both **pure LDP** and **ν-GLDP** settings.

### Pure LDP

```bash
python plot_finite_pure.py --k 10
python plot_finite_pure.py --k 20
python plot_finite_pure.py --k 100
```

### ν-GLDP

```bash
python plot_finite_GLDP.py --k 10
python plot_finite_GLDP.py --k 20
python plot_finite_GLDP.py --k 100
```

## Continuous Space Results

### 1D Laplace Mixture (Pure LDP)

```bash
python exp_1DLaplaceMix_pure.py --eps 0.1 --scale 1 --seed 1 --size 100
python exp_1DLaplaceMix_pure.py --eps 0.5 --scale 1 --seed 2 --size 100
python exp_1DLaplaceMix_pure.py --eps 1.0 --scale 1 --seed 3 --size 100
python exp_1DLaplaceMix_pure.py --eps 2.0 --scale 1 --seed 4 --size 100

python plot_1DLaplaceMix_pure.py
```

### 1D Laplace Mixture (ν-GLDP)

```bash
python exp_1DLaplaceMix_GLDP.py --nu 0.1 --scale 1 --seed 1 --size 100
python exp_1DLaplaceMix_GLDP.py --nu 0.5 --scale 1 --seed 2 --size 100
python exp_1DLaplaceMix_GLDP.py --nu 1.0 --scale 1 --seed 3 --size 100
python exp_1DLaplaceMix_GLDP.py --nu 2.0 --scale 1 --seed 4 --size 100

python plot_1DLaplaceMix_GLDP.py
```

### 2D Laplace Mixture (Pure LDP)

```bash
python exp_1DLaplaceMix_pure.py --eps 0.1 --scale 1 --seed 1 --size 100 --dim 2
python exp_1DLaplaceMix_pure.py --eps 0.5 --scale 1 --seed 2 --size 100 --dim 2
python exp_1DLaplaceMix_pure.py --eps 1.0 --scale 1 --seed 3 --size 100 --dim 2
python exp_1DLaplaceMix_pure.py --eps 2.0 --scale 1 --seed 4 --size 100 --dim 2

python plot_nDLaplaceMix_pure.py
```

## Output

Each plotting script generates a `.pdf` or `.png` file containing the reproduced figure. Make sure all required `.npy` files are saved before running the plotting scripts.
