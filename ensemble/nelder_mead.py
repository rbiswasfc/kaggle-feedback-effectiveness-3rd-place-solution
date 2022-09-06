import pandas as pd
import os
import numpy as np
import scipy as sp
import argparse
from sklearn.metrics import f1_score, log_loss
from scipy.optimize import minimize

parser = argparse.ArgumentParser(description='nelder-mead')
parser.add_argument('-custom', action='store_true',help='Patience')
parser.add_argument('-exclude', type=str, nargs='+', help='List of oof csvs', required=False)

args = parser.parse_args()

def get_score(y_true, y_pred):
    score = log_loss(y_true, y_pred, labels=[0,1,2])
    return round(score, 5)


PATH = './'
FILES = os.listdir(PATH)
OOF = np.sort([f for f in FILES if 'csv' in f])
if args.custom:
    OOF = [
        'exp1_oof_a-delv3-prod-8-folds.csv',
        #'exp2_oof_lf_0.5953.csv',
        # 'exp3_oof-lstm-multihead-attention.csv',
        # 'exp4_oof-resolved-lstm-multihead-attention.csv',
        # 'exp5_oof_a-prod-delv3-uda.csv',
        # 'exp6_oof_deb_xl_0.5788.csv',
        'exp7_oof_dexl_0.5728.csv',
        'exp8_oof_del_kd_0.5720.csv',
        # 'exp9_oof_delv3_0.5760.csv',
        'exp10_oof_delv3_0.5729.csv',
        #'exp11_oof_delv3_uda_0.5828.csv',
        # 'exp12_oof_delv3_gamma_0.5754.csv',
        #'exp14_oof_delv3_gamma_mixout.csv',
        #'exp15_oof_delv3_gamma_mixout.csv',
        'exp16_oof_delv3_10fold.csv',
        'exp17_oof_luke.csv',
        # 'exp102_oof_df_CV_5fold_.5804.csv',
        # 'exp201_oof_deb_l_0.5833.csv',
        # 'exp202_oof_deb_l_prompt_0.5809.csv',
        #'exp203_oof_deb_l_prompt_0.5784.csv',
        # 'exp204_oof_debv3_l_mnli_prompt.csv',
        #'exp205_oof_debv3_l_prompt_0.5754.csv',
        # 'exp206_oof_debv3_l_prompt.csv',
        # 'exp208_oof_debv2_xl_prompt.csv',
        'exp209_oof_debv3_l_10fold_prompt.csv',
        #'exp211_oof_funnel_l_prompt.csv',
        'exp212_oof_longformer_l_prompt_0.5833.csv',
        #'exp213_oof_deb_l_prompt_10folds.csv',
        'exp213a_oof_deb_l_prompt.csv',
        'exp214_oof_debv2-xl_prompt.csv',
        'exp215_oof_debv2_xl_prompt.csv',

    ]
    """
    OOF = [
        './level2/lgb_ensemble.csv',
        #'./level2/ens33_oof_lstm_9m.csv',
        './level2/ens42_meta_lstm.csv',
        #'./level2/ensemble_df.csv'
    ]
    """

excludes = args.exclude
if excludes:
    for exl in excludes:
        idx = np.where(OOF == exl)[0][0]
        OOF = np.delete(OOF, idx)


train_df = pd.read_csv('../datasets/feedback-prize-effectiveness/train.csv')
train_df["label"] = train_df["discourse_effectiveness"].map({"Ineffective": 0, "Adequate": 1, "Effective": 2})
train_df = train_df.sort_values(by='discourse_id').reset_index(drop=True)
TRUE = train_df.label.values

OOF_CSV = [pd.read_csv(PATH + k).sort_values(by='discourse_id', ascending=True).reset_index(drop=True) for k in OOF]

#base_df = pd.read_csv('oof_a-delv3-prod-8-folds.csv')
#print(base_df.shape)
#print(base_df['discourse_id'].nunique())
alloof = []
for i, tmpdf in enumerate(OOF_CSV):
    tmpdf.drop_duplicates(subset='discourse_id', inplace=True)
    #tmpdf.to_csv(f'tmp_{i}.csv', index=False)
    #print(tmpdf.shape)
    mpred = tmpdf[["Ineffective", "Adequate", "Effective"]].values
    alloof.append(mpred)



def min_func(K):
    ypredtrain = 0
    for a in range(len(alloof)):
        ypredtrain += K[a] * alloof[a]
    return get_score(TRUE, ypredtrain)


res = minimize(min_func, [1 / len(alloof)] * len(alloof), method='Nelder-Mead', tol=1e-6)
K = res.x
# Override wts here
"""
K = [0.11, # exp7 - dexl
     0.07, # exp8 - kd
     0.11, # exp10 - debv3-l 8 fold
     0.10, # exp16 - debv3-l 10 fold
     0.20, # exp209 - debv3-l
     0.11, # exp212 - lf
     0.24, # exp213 - deb-l
     0.06  # exp214 - v2-xl
     ]
"""
#K = [0.4, 0.6]
print(K)

ypredtrain = 0

oof_files = []
oof_wts = []

for a in range(len(alloof)):
    print(OOF[a], K[a])
    oof_files.append(OOF[a])
    oof_wts.append(K[a])
    ypredtrain += K[a] * alloof[a]



score = get_score(TRUE, ypredtrain)
print(f'Score: {score}')
print(f'Wt sum: {np.sum(K)}')
# recheck values

ens_df = pd.DataFrame(data={'oof_file': oof_files, 'wt': oof_wts})
ens_df = ens_df.sort_values(by='wt', ascending=False).reset_index(drop=True)
print(ens_df)

ensemble_df = pd.DataFrame()
ensemble_df['discourse_id'] = train_df['discourse_id']
ensemble_df['label'] = TRUE
ensemble_df = pd.concat([ensemble_df, pd.DataFrame(ypredtrain)], axis=1)
ensemble_df.rename(columns={0: "Ineffective", 1:"Adequate", 2:"Effective"}, inplace=True)
os.makedirs('./level2', exist_ok=True)
#ensemble_df.to_csv('./level2/ensemble_df.csv', index=False)