from train_NAVAR import train_NAVAR
from evaluate import calculate_AUROC, dream_file_to_causal_matrix
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Train NAVAR on DREAM3 or CauseMe data')
parser.add_argument('--experiment', metavar='experiment', type=str, help='name of the experiment (e')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--lstm', action='store_true')
args = parser.parse_args()
experiment = args.experiment
lstm = args.lstm

if experiment == 'ecoli1':
    if lstm:
        lambda1 = 0.26025947107502856
        batch_size = 46
        wd = 1.4961159190152877e-05
        hidden_nodes = 10
        learning_rate = 0.002
        hl = 1
    else:
        lambda1 = 0.18831046469152488
        batch_size = 128
        wd = 0.00011149985622397591
        hidden_nodes = 10
        learning_rate = 0.0005
        hl = 1
elif experiment == 'ecoli2':
    if lstm:
        lambda1 = 0.1981487034148272
        batch_size = 46
        wd = 2.2672318420640924e-05
        hidden_nodes = 10
        learning_rate = 0.002
        hl = 1
    else:
        lambda1 = 0.20109211949576583
        batch_size = 32
        wd = 0.00017095046240606562
        hidden_nodes = 10
        learning_rate = 0.001
        hl = 1
elif experiment == 'yeast1':
    if lstm:
        lambda1 = 0.17687833718877716
        batch_size = 46
        wd = 2.6310335018015632e-05
        hidden_nodes = 10
        learning_rate = 0.002
        hl = 1
    else:
        lambda1 = 0.26967055660779576
        batch_size = 16
        wd = 0.00014248297214870847
        hidden_nodes = 10
        learning_rate = 0.002
        hl = 2
elif experiment == 'yeast2':
    if lstm:
        lambda1 = 0.2762570334308751
        batch_size = 46
        wd = 1.8383563225123858e-05
        hidden_nodes = 10
        learning_rate = 0.002
        hl = 1
    else:
        lambda1 = 0.15625106104737252
        batch_size = 256
        wd = 0.00020125408829060009
        hidden_nodes = 10
        learning_rate = 0.0002
        hl = 1
elif experiment == 'yeast3':
    if lstm:
        lambda1 = 0.21806234949103717
        batch_size = 46
        wd = 6.412088523344369e-06
        hidden_nodes = 10
        learning_rate = 0.002
        hl = 1
    else:
        lambda1 = 0.15592131669202652
        batch_size = 16
        wd = 0.0001644367616763493
        hidden_nodes = 10
        learning_rate = 0.0002
        hl = 1

# load the data
file = f'experiments/{experiment}.tsv'
ground_truth_file = f'experiments/{experiment}_gt.txt'
data = pd.read_csv(file, sep='\t')
data = data.values[:, 1:]
epochs = 5000
maxlags = 21 if lstm else 2

# start training
print(f"Starting training on the data from experiment {experiment}, training for {epochs} iterations.")
score_matrix, _, _ = train_NAVAR(data, maxlags=maxlags, hidden_nodes=hidden_nodes, dropout=0, epochs=epochs,
                                 learning_rate=learning_rate, batch_size=batch_size, lambda1=lambda1,
                                 val_proportion=0.0, weight_decay=wd, check_every=500, hidden_layers=hl, normalize=True,
                                 split_timeseries=21, lstm=lstm)
# evaluate
print('Done training!')
if args.evaluate:
    ground_truth_matrix = dream_file_to_causal_matrix(ground_truth_file)
    AUROC = calculate_AUROC(score_matrix, ground_truth_matrix, ignore_self_links=True)
    print(f"The AUROC of this model on experiment {experiment} is: {AUROC}")
