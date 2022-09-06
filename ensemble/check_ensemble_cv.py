import os
import argparse
import pandas as pd
from sklearn.metrics import f1_score, log_loss

def get_score(y_true, y_pred):
    score = log_loss(y_true, y_pred, labels=[0,1,2])
    return round(score, 5)

parser = argparse.ArgumentParser(description='Process pytorch params.')

parser.add_argument('-train_file', help='Algorithm to use')
parser.add_argument('-csv', type=str, help='List of oof csvs', required=True)
parser.add_argument('-fold', type=int, help='List of oof csvs', required=True)

args = parser.parse_args()


oof_df = pd.read_csv(args.csv)
fold_df = pd.read_parquet(args.train_file)
df = pd.read_csv("../datasets/feedback-prize-effectiveness/train.csv")
fold_df = pd.merge(df, fold_df, on="essay_id", how="left")
oof_df = oof_df.merge(fold_df[['discourse_id', 'kfold']], how='left')
oof_fold_df = oof_df[oof_df['kfold']==args.fold]

truth = oof_fold_df['label'].values
preds = oof_fold_df[["Ineffective", "Adequate", "Effective"]].values

score = get_score(truth, preds)
print(f'Fold {args.fold} Score: {score}')


