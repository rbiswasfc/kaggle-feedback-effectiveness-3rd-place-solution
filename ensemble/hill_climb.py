#!/usr/bin/env python

import pandas as pd, numpy as np, os

from sklearn.metrics import f1_score, log_loss
import argparse

parser = argparse.ArgumentParser(description='hill climbing')
parser.add_argument('-tol', type=float, default=0.0003, help='Tolerance. Decrease to try more models')
parser.add_argument('-pat', type=int, default=10, help='Patience')
parser.add_argument('-custom', action='store_true', help='Patience')
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
        #'exp1_oof_a-delv3-prod-8-folds.csv',
        #'exp2_oof_lf_0.5953.csv',
        #'exp3_oof-lstm-multihead-attention.csv',
        #'exp4_oof-resolved-lstm-multihead-attention.csv',
        #'exp5_oof_a-prod-delv3-uda.csv',
        #'exp6_oof_deb_xl_0.5788.csv',
        'exp7_oof_dexl_0.5728.csv',
        #'exp8_oof_del_kd_0.5720.csv',
        #'exp9_oof_delv3_0.5760.csv',
        #'exp10_oof_delv3_0.5729.csv',
        'exp11_oof_delv3_uda_0.5828.csv',
        #'exp12_oof_delv3_gamma_0.5754.csv',
        #'exp14_oof_delv3_gamma_mixout.csv',
        #'exp15_oof_delv3_gamma_mixout.csv',
        'exp16_oof_delv3_10fold.csv',
        #'exp102_oof_df_CV_5fold_.5804.csv',
        #'exp201_oof_deb_l_0.5833.csv',
        #'exp202_oof_deb_l_prompt_0.5809.csv',
        'exp203_oof_deb_l_prompt_0.5784.csv',
        #'exp204_oof_debv3_l_mnli_prompt.csv',
        #'exp205_oof_debv3_l_prompt_0.5754.csv',
        #'exp206_oof_debv3_l_prompt.csv',
        #'exp208_oof_debv2_xl_prompt.csv',
        'exp209_oof_debv3_l_10fold_prompt.csv',
        #'exp211_oof_funnel_l_prompt.csv',
        'exp212_oof_longformer_l_prompt_0.5833.csv',
    ]
    """
    OOF = [
        './level2/ensemble_df.csv',
        './level2/lgb_ensemble.csv',
        #'./level2/ens33_oof_lstm_9m.csv'
        ]
    """
excludes = args.exclude
if excludes:
    for exl in excludes:
        print(exl)
        idx = np.where(OOF == exl)[0][0]
        OOF = np.delete(OOF, idx)

print(OOF)
OOF_CSV = []
OOF_CSV = [pd.read_csv(PATH + k).sort_values(by='discourse_id', ascending=True).reset_index(drop=True) for k in OOF]

print('We have %i oof files...' % len(OOF))

# 3 labels
x = np.zeros((len(OOF_CSV[0]), 3, len(OOF)))
#print(x.shape)
for k in range(len(OOF)):
    oof_df = OOF_CSV[k]
    oof_df.drop_duplicates(subset='discourse_id', inplace=True)
    values = oof_df[["Ineffective", "Adequate", "Effective"]].values
    x[:, :, k] = values

#print(x.shape)
try:
	train_df = pd.read_csv('../datasets/feedback-prize-effectiveness/train.csv')
except:
	train_df = pd.read_csv("C:\\Users\\mehta\\Desktop\\kaggle\\feedback-prize-effectiveness\\train.csv")
train_df["label"] = train_df["discourse_effectiveness"].map({"Ineffective": 0, "Adequate": 1, "Effective": 2})
train_df = train_df.sort_values(by='discourse_id').reset_index(drop=True)
TRUE = train_df.label.values


all = []
for k in range(x.shape[2]):
    score = get_score(TRUE, x[:, :, k])
    all.append(score)
    print('Model %i has OOF score = %.4f' % (k, score))

m = [np.argmin(all)]
w = []

old = np.min(all)

RES = 200
PATIENCE = args.pat
TOL = args.tol
DUPLICATES = False

print('Ensemble score = %.4f by beginning with model %i' % (old, m[0]))
print()

for kk in range(len(OOF)):

    # BUILD CURRENT ENSEMBLE
    md = x[:, :, m[0]]
    for i, k in enumerate(m[1:]):
        md = w[i] * x[:, :, k] + (1 - w[i]) * md

    # FIND MODEL TO ADD
    mx = 1000
    mx_k = 0
    mx_w = 0
    print('Searching for best model to add... ')

    # TRY ADDING EACH MODEL
    for k in range(x.shape[2]):
        print(k, ', ', end='')
        #print(f'k= {k}, m={m}')
        if not DUPLICATES and (k in m): continue
        # EVALUATE ADDING MODEL K WITH WEIGHTS W
        bst_j = 0
        bst = 10000
        ct = 0
        for j in range(RES):
            #print(f'Md shape: {md.shape}')
            tmp = j / RES * x[:, :,  k] + (1 - j / RES) * md
            #print(f'tmp shape: {tmp.shape}')
            score = get_score(TRUE, tmp)
            #print(f'Score: {score}')
            if score < bst:
                bst = score
                bst_j = j / RES
            else:
                ct += 1
            if ct > PATIENCE: break
        #print(f'bst: {bst} mx: {mx}')
        if bst < mx:
            mx = bst
            mx_k = k
            mx_w = bst_j

    # STOP IF INCREASE IS LESS THAN TOL
    inc = old - mx
    #print(f'inc: {inc} tol: {TOL}')
    if inc <= TOL:
        print()
        print('No decrease. Stopping.')
        break

    # DISPLAY RESULTS
    print()  # print(kk,mx,mx_k,mx_w,'%.5f'%inc)
    print('Ensemble score = %.4f after adding model %i with weight %.3f. Decrease of %.4f' % (mx, mx_k, mx_w, inc))
    print()

    old = mx
    m.append(mx_k)
    w.append(mx_w)

print(f'We are using {len(m)} models: {m}')
print('with weights', w)
print('and achieve a score of = %.4f' % old)

# base model preds
md = x[:,:,  m[0]]

w_ = []
for i, k in enumerate(m[1:]):
    #print(f'wt: {w[i]} {1-w[i]}')
    w_.append(1-w[i])
    md = w[i] * x[:, :, k] + (1 - w[i]) * md


score = get_score(TRUE, md)
ensemble_df = pd.DataFrame()
ensemble_df['discourse_id'] = train_df['discourse_id']
ensemble_df['label'] = TRUE
ensemble_df = pd.concat([ensemble_df, pd.DataFrame(md)], axis=1)
ensemble_df.rename(columns={0: "Ineffective", 1:"Adequate", 2:"Effective"}, inplace=True)
os.makedirs('./level2', exist_ok=True)
#ensemble_df.to_csv('./level2/ensemble_df.csv', index=False)


#print('--' * 5)
#print(f'w:{w} w_: {w_}')
print('----'*5)

for i, row in enumerate(m):
    print(f"'{OOF[row]}',")

print('----'*5)
print('-Weights-')
wt_sum = 0
final_wts = []
for i, row in enumerate(m):
    if i==0:
        wts = w_[i]
        for wt_ in w_[1:]:
            #print(wt_)
            wts *= wt_
    else:
        wts = w[i-1]
        for wt_ in w_[i:]:
            #print(wt_)
            wts *= wt_
    wts = np.round(wts, 5)
    wt_sum += wts
    print(f"{OOF[row]} wt: {wts}")
    final_wts.append(wts)
print(f'Score: {score}')
print(f'wts: {final_wts}')
print(f'Wt sum: {wt_sum}')