import os

folds = 8
trials = 50
output_dir = 'lgb_ens35'
train_file = 'level2/train_ens35.csv'

for fold in [0, 1, 2, 3, 4, 5, 6,7]:
    #os.system(f"python train_models.py -train_file {train_file} -model_dir {output_dir} -run_folds {fold} -optuna -n_trials {trials} -add_features")
    os.system(f"python train_models.py -train_file {train_file} -model_dir {output_dir} -run_folds {fold} -optuna -n_trials {trials}")
