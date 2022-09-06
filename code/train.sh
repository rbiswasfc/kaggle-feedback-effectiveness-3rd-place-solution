#!/usr/bin/env bash
python tools/process_data.py --data_dir ../datasets/feedback-prize-effectiveness --save_dir ../datasets/processed/

python tools/fpe_span_mlm_delv3_prod.py --config_path code/configs/mlm/mlm_config_delv3_prod.json
python tools/fpe_span_mlm_dexl.py --config_path code/configs/mlm/mlm_config_dexl.json
python tools/fpe_span_mlm_luke_prod.py --config_path code/configs/mlm/mlm_config_luke_prod.json
python tools/fpe_span_mlm_del.py --config_path code/configs/mlm/mlm_config_del.json


for fold in 0 1 2 3 4 5 6 7 8 9
do
    python train_delv3.py --config_path code/configs/delv3/config_delv3.json --fold $fold
done

for i_seed in 0 1
do
    python train_delv3_all.py --config_path code/configs/delv3/config_delv3_all.json
done

for fold in 0 1 2 3 4 5 6 7
do
    python train_dexl.py --config_path code/configs/dexl/exp_config_dexl.json --fold $fold
done

for i_seed in 0 1
do
    python train_dexl_all.py --config_path code/configs/dexl/exp_config_dexl_all.json
done

for fold in 0 1 2 3 4 5 6 7 8 9
    python train_luke_multitask.py --config_path code/configs/luke/config_luke_multitask.json --fold $fold
done 

for i_seed in 0 1
do
    python train_luke_multitask_all.py --config_path code/configs/luke/config_luke_multitask_full_train.json
done

for i_seed in 0 1
do
    python train_del_kd_all.py --config_path code/configs/del/exp_config_del_kd_all.json
done


python tools/fpe_span_mlm_del_prompt.py --config_path configs/mlm/mlm_config_del_prompt.json
for fold in 0 1 2 3 4 5 6 7 8 9
do 
	  python train_del_prompt.py --config_path configs/del_prompt/exp_config_del.json --fold $fold
done

python tools/fpe_span_mlm_delv3_prompt.py --config_path configs/mlm/mlm_config_delv3_prompt.json
for fold in 0 1 2 3 4 5 6 7 8 9
do
	  python train_fast_prompt.py --config_path configs/fast_prompt/exp_config_fast.json --fold $fold
done

python tools/fpe_span_mlm_longformer_prompt.py configs/longformer_prompt/exp_config_longformer.json
for fold in 0 1 2 3 4 5 6 7
do
	  python train_longformer_prompt.py --config_path configs/longformer_prompt/exp_config_longformer.json --fold $fold
done


for fold in 0 1 2 3 4 5 6 7
do
	  python train_meta_rb.py --config_path configs/meta/config_meta_rb.json --fold $fold
done

for fold in 0 1 2 3 4 5 6 7
do
	  python hm-meta-lgb-script.py --config_path configs/meta/config_meta_lgb_hm.json --fold $fold
done
