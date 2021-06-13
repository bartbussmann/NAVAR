# NAVAR

Code for Neural Additive Vector Autoregression Models for Causal Discovery in Time Series Data ([Paper](paper/Neural_Additive_Vector_Autoregression_Models_for_Causal_Discovery_in_Time_Series.pdf), [Supplementary Material](paper/Supplementary_Material.pdf)).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements. 

```bash
git clone https://github.com/bartbussmann/NAVAR
cd NAVAR
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

### Run on your own data
In order to run NAVAR on your own dataset, save your data in a CSV file of T rows x N columns, where T is the number of time steps and N is the number of variables. Then, use the following command:

```bash
python run_NAVAR.py --filename my_data.csv
```

where you replace my_data.csv with the path to your own csv-file. If you want to run NAVAR with specific hyperparameters, run it with the following flags:

```bash
python run_NAVAR.py --filename my_data.csv --maxlags 5 --hidden_nodes 10 --hidden_layers 1 --epochs 2000 --batch_size 32 --sparsity_penalty 0.1 --weight_decay 1e-4 --dropout 0.0 --learning_rate 3e-4 --validation_proportion 0.2
```

