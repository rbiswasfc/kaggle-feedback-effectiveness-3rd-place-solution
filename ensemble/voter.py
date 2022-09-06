#!/usr/bin/env python

import pandas as pd, numpy as np, os

from sklearn.metrics import f1_score, log_loss
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description='hill climbing')
parser.add_argument('-custom', action='store_true', help='Patience')
parser.add_argument('-verbose', action='store_true', help='Patience')
parser.add_argument('-debug', action='store_true', help='Patience')

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
        'exp7_oof_dexl_0.5728.csv',
        #'oof_deb_l_0.5833.csv',
        #'oof-resolved-lstm-multihead-attention.csv',
        #'oof_lf_0.5953.csv',
    ]

print(OOF)
OOF_CSV = []
if args.debug:
    OOF_CSV = [pd.read_csv(PATH + k, nrows=100).sort_values(by='discourse_id', ascending=True).reset_index(drop=True) for k in OOF]
else:
    OOF_CSV = [pd.read_csv(PATH + k).sort_values(by='discourse_id', ascending=True).reset_index(drop=True) for k in OOF]

train_df = pd.read_csv('../datasets/feedback-prize-effectiveness/train.csv')
train_df["label"] = train_df["discourse_effectiveness"].map({"Ineffective": 0, "Adequate": 1, "Effective": 2})
train_df = train_df.sort_values(by='discourse_id', ascending=True).reset_index(drop=True)

if args.debug:
    train_df = pd.merge(train_df, OOF_CSV[0][['discourse_id']], how='inner')
    print(train_df.shape)

TRUE = train_df.label.values

def add_label(row):
    cols = ['Adequate', 'Effective', 'Ineffective']
    prob = np.array([row['Adequate'],row['Effective'], row['Ineffective']])
    #print(prob)
    # Top n probs
    n = 2
    indices = (-prob).argsort()[:n]
    #print(indices)
    return cols[indices[0]], cols[indices[1]]

for oof in OOF_CSV:
    oof[['preds0', 'preds1']] = oof.apply(lambda x: add_label(x), axis=1, result_type='expand')

print('We have %i oof files...' % len(OOF))

oofs = OOF_CSV


def ensemble_df(oofs):
    verbose = args.verbose
    prediction_col = 'preds0'
    prediction_col1 = 'preds1'
    final_predictions = []
    for i, id in enumerate(oofs[0]['discourse_id'].values):
        preds = []
        predictions = []
        for j in range(len(oofs)):
            preds.append(oofs[j][oofs[j]['discourse_id'] == id][prediction_col].values)
            #print(preds)
            if not pd.isna(preds[j]):
                predictions.append(preds[j][0])

        if len(predictions) > 0:
            x = Counter(predictions)
            if verbose:
                print(f'All predictions {x}')
                print(f'Most common: {x.most_common(1)[0]}')
            # Select majority
            prediction0, vote_count0 = x.most_common(1)[0]
            prediction1, vote_count1 = x.most_common(2)[-1]
            level1_pred = prediction0
            if verbose:
                print(f'vote counts: {vote_count0}  {vote_count1} {len(x)}')
            # diff betweem most common and common should be greater
            if len(x) == 1:
                final_pred = prediction0
                if verbose:
                    print(f'Level1 final pred: {final_pred}')
            elif len(x) > 1 and vote_count0 - vote_count1 > 1:
                final_pred = prediction0
                if verbose:
                    print(f'Level1 final pred: {final_pred}')
            # If no majority add next best preds to vote
            else:
                if verbose:
                    print('Level2')
                    print(predictions)
                preds = []
                for j in range(len(oofs)):
                    preds.append(oofs[j][oofs[j]['discourse_id'] == id][prediction_col1].values)
                    # print(preds)
                    if not pd.isna(preds[j]):
                        predictions.append(preds[j][0])
                if len(predictions) > 0:
                    x = Counter(predictions)
                    if verbose:
                        print(f'L2: All predictions {x}')
                        print(f'Most common: {x.most_common(1)[0]}')
                    # Select majority
                    prediction0, vote_count0 = x.most_common(1)[0]
                    prediction1, vote_count1 = x.most_common(2)[-1]
                    if vote_count0 - vote_count1 > 1:
                        final_pred = prediction0
                    else:
                        final_pred = level1_pred
                    if verbose:
                        print(f'Level2 final pred: {final_pred}')
            final_predictions.append(final_pred)
        else:
            pass
        if verbose:
            print(f'id: {id} pred: {final_pred}')
            print('--' * 5)
    return final_predictions


def proc_csv(row):
    """
    To convert back to probs for logloss metric
    :param row:
    :return:
    """
    if row['preds'] == 'Adequate':
        return 0.9, 0.05, 0.05
    elif row['preds'] == 'Effective':
        return 0.05, 0.9, 0.05
    elif row['preds'] == 'Ineffective':
        return 0.05, 0.05, 0.9

final_preds = ensemble_df(oofs)
out_df = pd.DataFrame({'discourse_id': oofs[0]['discourse_id'], 'preds': final_preds})

from sklearn.metrics import f1_score
#out_df = pd.DataFrame({'discourse_id': oofs[0]['discourse_id'], 'preds': OOF_CSV[0]['preds0']})
out_df[['Adequate', 'Effective', 'Ineffective']] = out_df.apply(lambda x: proc_csv(x), axis=1, result_type='expand')

#out_df = OOF_CSV[0]
out_df["label"] = out_df["preds"].map({"Ineffective": 0, "Adequate": 1, "Effective": 2})
score = get_score(TRUE, out_df[["Ineffective", "Adequate", "Effective"]].values)
print(f'Score: {score}')

f1 = f1_score(TRUE, out_df['label'].values, average='micro')
print(f'f1 score: {f1}')
out_df.to_csv('level2/ensemble_vote.csv', index=False)
print(f'Generated: level2/ensemble_vote.csv')