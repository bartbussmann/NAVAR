from train_NAVAR import train_NAVAR
import numpy as np
import argparse
import zipfile
import json
import bz2


parser = argparse.ArgumentParser(description='Train NAVAR on CauseMe data')
parser.add_argument('--experiment', metavar='experiment', type=str, help='name of the experiment (e')
parser.add_argument('--method_sha', metavar='method_sha', type=str, help='name of the experiment (e')

args = parser.parse_args()
experiment = args.experiment
method_sha = args.method_sha
if experiment == 'nonlinear-VAR_N-3_T-300':
    lambda1 = 0.1344
    batch_size = 64
    wd = 2.903e-3
    hidden_nodes = 32
    learning_rate = 0.00005
    hl = 1
    maxlags = 5

elif experiment == 'nonlinear-VAR_N-5_T-300':
    lambda1 = 0.1596
    batch_size = 64
    wd = 2.420e-3
    hidden_nodes = 16
    learning_rate = 0.0001
    hl = 1
    maxlags = 5

elif experiment == 'nonlinear-VAR_N-10_T-300':
    lambda1 = 0.2014
    batch_size = 64
    wd = 8.557e-3
    hidden_nodes = 128
    learning_rate = 0.0005
    hl = 1
    maxlags = 5


elif experiment == 'nonlinear-VAR_N-20_T-300':
    lambda1 = 0.2434
    batch_size = 64
    wd = 4.508e-3
    hidden_nodes = 32
    learning_rate = 0.0002
    hl = 1
    maxlags = 5

elif experiment == 'TestCLIM_N-40_T-250':
    lambda1 = 0.3924
    batch_size = 16
    wd = 4.322e-3
    hidden_nodes = 32
    learning_rate = 0.0002
    hl = 1
    maxlags = 2

elif experiment == 'TestWEATH_N-20_T-2000':
    lambda1 = 0.0560
    batch_size = 64
    wd = 4.903e-3
    hidden_nodes = 32
    learning_rate = 0.0001
    hl = 1
    maxlags = 5


elif experiment == 'river-runoff_N-12_T-4600':
    lambda1 = 0.1760
    batch_size = 128
    wd = 4.749e-7
    hidden_nodes = 64
    learning_rate = 0.0002
    hl = 1
    maxlags = 5

# prepare results file
results = {}
results["method_sha"] = method_sha
results["parameter_values"] = f'maxlags: {maxlags}'
results['model'] = "NAVAR"
results['experiment'] = results['model'] + '_' + experiment
results_file = f'results/{experiment}.json.bz2'
scores = []

# load the data
file = f'experiments/{experiment}.zip'
with zipfile.ZipFile(file, "r") as zip_ref:
    datasets = sorted(zip_ref.namelist())
    for dataset in datasets:
        print(f"Training NAVAR on: {dataset}")
        data = np.loadtxt(zip_ref.open(dataset))
        # start training NAVAR
        score_matrix, _, _ = train_NAVAR(data, maxlags=maxlags, hidden_nodes=hidden_nodes, dropout=0, epochs=2000,
                                         learning_rate=learning_rate, batch_size=batch_size, lambda1=lambda1,
                                         val_proportion=0.0, weight_decay=wd, check_every=500, hidden_layers=hl,
                                         normalize=True, split_timeseries=False)
        scores.append(score_matrix.flatten())

# Save data
print('Writing results ...')
results['scores'] = np.array(scores).tolist()
results_json = bytes(json.dumps(results), encoding='latin1')
with bz2.BZ2File(results_file, 'w') as mybz2:
    mybz2.write(results_json)