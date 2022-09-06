### 1. CREATE FOLDS: 
Follow the steps in README.md for data setup first. You can then use the commands below to create a new folds file for training. 
* `cd code`
* `python tools/process_data.py --data_dir ../datasets/feedback-prize-effectiveness --save_dir ../datasets/processed/`

### 2. MODEL BUILD: 

You can specify the fold file created using step 1 in the config file located in `code/configs/` depending on the architecture you want to train. 
You can then specify that config file in the command line below. 

* `cd code`
* `python train_fast.py --config_path ./configs/exp_config_fast.json --fold 0 --use_wandb`

### 3. MODEL INFERENCE: 

* `cd code `
* `python predict_fast.py --config_path ./configs/exp_config_fast.json --model_path ./models/<trained_model_path>`