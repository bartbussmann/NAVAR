# NAVAR

Code for Neural Additive Vector Autoregression Models for Causal Discovery in Time Series Data

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements. 

```bash
pip install -r requirements.txt
```

Python >= 3.5 is required.
## Usage

### DREAM3 experiments

In order to reproduce the experiments on the DREAM3 dataset use the following command:

```bash
python run_dream_experiments.py --experiment experiment_name --evaluate --lstm
```

The experiment names are `ecoli1`, `ecoli2`, `yeast1`, `yeast2`, and `yeast3`. The `--lstm` flag indicated whether you want to use the lstm model or not. The `--evaluate` flag makes sure the score matrix is compared to the ground truth, and the AUROC is printed. 

### CauseMe experiments
In order to reproduce the experiments from the CauseMe platform use the following command:
```bash
python run_causeme_experiments.py --experiment experiment_name
```

An example of an experiment name is `nonlinear-VAR_N-5_T-300`. This will produce a file in the results folder with the experiment name that contains the score matrices. The score matrix can only be evaluated by the CauseMe platform. If one wants to do so, an account should be registered and a method_SHA should be obtained. Then, you can reproduce the experiment (for instance nonlinear-VAR_N-5_T-300) by running:
```bash
python run_causeme_experiments.py --experiment nonlinear-VAR_N-5_T-300 --method_sha XXXXXX
```

