import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def _get_score(y_true, y_pred):
    score = log_loss(y_true, y_pred)
    return round(score, 4)


def get_score(df):
    return _get_score(df["label"].values, df[["Ineffective", "Adequate", "Effective"]].values)


def human_format(num, suffix=""):
    """
    human readable format for numbers
    """
    for unit in ["", "k", "m", "g", "t", "p", "e", "z"]:
        if num < 1000.0:
            return "{:.1f}{}{}".format(num, unit, suffix)
        num /= 1000.0
    return "{:.1f}{}{}".format(num, "Y", suffix)


def get_oof_report(oof_df, train_df):
    """
    helper function for error analysis

    :param oof_df: oof dataframe from k-fold model checkpoints
    :type oof_df: pd.DataFrame
    :param train_df: training dataframe
    :type train_df: pd.DataFrame
    """
    print("=="*40)
    print("|Running profiler...")
    oof_df = pd.merge(oof_df, train_df, on="discourse_id", how="left")

    oof_df["loss"] = oof_df[["label", "Ineffective", "Adequate", "Effective"]].apply(
        lambda x: -np.log(x[int(x[0]+1)]), axis=1)

    print("=="*40)
    print("|Loss stats...")
    max_l = round(oof_df['loss'].max(), 4)
    ave_l = round(oof_df['loss'].mean(), 4)
    std_l = round(oof_df['loss'].std(), 4)
    rp = f"|Max:{max_l: <4}| Ave:{ave_l: <4}| SD:{std_l: <4}"
    print(rp)

    print("=="*40)
    print("|Model performance breakdown as per discourse effectiveness...")
    for gn, tmp_df in oof_df.groupby("discourse_effectiveness"):
        ll_mean = round(tmp_df[gn].apply(lambda x: -np.log(x)).mean(), 4)
        total = human_format(ll_mean*len(tmp_df))
        rp = f"|{gn:<15}:{ll_mean: <4}| count: {len(tmp_df): <6} | sum_loss: {total:<10}|"
        print(rp)
    print("=="*40)

    print("|Model performance breakdown as per discourse type...")
    for gn, tmp_df in oof_df.groupby("discourse_type"):
        s = round(get_score(tmp_df), 4)
        total = human_format(s*len(tmp_df))
        rp = f"|{gn:<25}:{s: <4}| count: {len(tmp_df): <6} | sum_loss: {total:<10}|"
        print(rp)
    print("=="*40)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', type=str, required=True)
    ap.add_argument('--oof_path', type=str, required=True)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_path)
    oof_df = pd.read_csv(args.oof_path)

    get_oof_report(oof_df, train_df)
