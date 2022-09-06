#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:22:27 2021

Train a GBM/Tabnet
Support for Optuna for hyper params search
RFE - Recursive feature eliminiation

@author: trushk
"""

import pandas as pd
import numpy as np
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, auc, log_loss
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
import os
import json
import gc
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge
import argparse
import random
import pickle
from sklearn.decomposition import PCA
from collections import OrderedDict, defaultdict
# from joblib import dump, load
from pickle import dump
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE, RFECV
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from shaphypetune import BoostSearch, BoostRFE, BoostBoruta
from sklearn.preprocessing import MinMaxScaler
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.decomposition import TruncatedSVD
from math import gamma
from zoofs import GeneticOptimization
# from kaggler.model import AutoLGB
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric

parser = argparse.ArgumentParser(description='Process pytorch params.')

parser.add_argument('-algo', type=str, choices=['lgb', 'xgb', 'svm', 'tabnet', 'ridge', 'rf'], default='lgb',
                    help='Algorithm to use')
parser.add_argument('-use_autolgb', action='store_true', help='Algorithm to use')
parser.add_argument('-model_dir', type=str, default='test', help='output dir to store results')
parser.add_argument('-batch_size', type=int, default=32, help='batch_size')
parser.add_argument('-gpu', type=int, default=[0, 1], nargs='+', help='use gpu')
parser.add_argument('-pca', type=int, help='PCA')
parser.add_argument('-debug', type=int, help='debug mode. small batch size')
parser.add_argument('-optuna', action='store_true', help='Use dropout layer')
parser.add_argument('-device', type=str, default='cuda', help='Use dropout layer')
parser.add_argument('-train_file', type=str, default='folds.csv', help='train file')
parser.add_argument('-n_trials', type=int, default=10, help='number of optuna trials')
parser.add_argument('-fs', type=str, help='Feature selection. RFE/Eli5')
parser.add_argument('-n_features', type=int, default=30, help='feature selection')
parser.add_argument('-step', type=float, default=1, help='Step size for RFE')
parser.add_argument('-features', type=str, help='optional pkl of features names (list) to use')
parser.add_argument('-epochs', type=int, default=1000, help='feature slection')
parser.add_argument('-run_folds', type=int, nargs='+', help='Run targetted folds')
parser.add_argument('-lr', type=float, help='learning rate')
parser.add_argument('-ga', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-svd', type=float, help='Recursive Feature Elimination')
parser.add_argument('-add_features', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-add_misc', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-full_data', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-full_fe', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-drop_outlier', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-only_outlier', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-kfold_knn', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-kmeans_agg', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-infer', action='store_true', help='Recursive Feature Elimination')
parser.add_argument('-infer_dir', type=str, default='test', help='output dir to store results')
parser.add_argument('-submit', action='store_true', help='Recursive Feature Elimination')

args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)

ID_COL = 'essay_id'
TARGET_COL = 'discourse_effectiveness'
DEVICE = torch.device(args.device)

preds = 0
num_folds = 5
random_state = 1234
val_aucs = []

# Create models dir in folder
model_folder = '../output'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

# Create dir with lgb name + timestamp
op_folder = model_folder + '/' + args.model_dir + '/'
if not os.path.exists(op_folder):
    os.mkdir(op_folder)

drop_columns = [ID_COL, 'discourse_id']


## Recursive Feature Elimination
def feature_selection(data):
    for fold in range(1):
        print(f'Fold {fold}')

        seed = 42
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            "num_class": 3,
            'metric': 'multiclass',  # multi_logloss #multiclass
            'learning_rate': 0.008,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'verbosity': 1,
            "max_depth": -1,
            "num_leaves": 15,
            "max_bin": 32,
            'is_unbalance': 'true'
        }


        if args.kfold_knn:
            print('Using Kfold KNN')
            indexes = np.arange(5).astype(int)
            indexes = np.delete(indexes, obj=fold, axis=0)
            with open('values.pkl', 'rb') as f:
                values = pickle.load(f)
            indexes = np.r_[values[indexes[0]], values[indexes[1]], values[indexes[2]], values[indexes[3]]]
            train_data = data.loc[data.time_id.isin(indexes)]
            valid_data = data.loc[data.time_id.isin(values[fold])]
            train_idx = indexes
            valid_idx = values[fold]
        else:
            train_data = data[data['kfold'] != fold].reset_index(drop=True)
            valid_data = data[data['kfold'] == fold].reset_index(drop=True)

            train_idx = data[data['kfold'] != fold].index.values
            valid_idx = data[data['kfold'] == fold].index.values

        y_train = train_data[TARGET_COL].values
        y_val = valid_data[TARGET_COL].values



        # Remove columns from dataframe
        if not args.features:
            train_data = train_data.drop(drop_columns, axis=1)
            valid_data = valid_data.drop(drop_columns, axis=1)

        features = list(train_data.columns)
        if not args.add_features:
            features.remove(TARGET_COL)
            #features.remove(ID_COL)
            features.remove('kfold')

        X_train = train_data[features]
        X_val = valid_data[features]

        # Add FE here
        if args.add_features:
            print('Doing Feature enginering on data')
            print(f'Previous data shape: {X_train.shape}')
            features = list(X_train)
            features.remove(TARGET_COL)
            features.remove('kfold')

            X_train = X_train[features]
            X_val = X_val[features]
            print(f'After FE data shape: {X_train.shape}')

        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)

        if args.svd:
            svd = TruncatedSVD(n_components=args.svd, n_iter=7, random_state=42)
            # X_train = X_train.fillna(X_train.mean())
            # X_val = X_val.fillna(X_train.mean())

            X_train = svd.fit_transform(X_train)
            X_val = svd.transform(X_val)

        print(f'Data shape: {X_train.shape}')

        # FIXME for debug only
        # X_train = X_train.iloc[:,0:40]
        from shaphypetune import BoostSearch, BoostBoruta, BoostRFE, BoostRFA

        if args.fs == 'rfe':
            """
            model = BoostRFE(
                lgbm.LGBMClassifier(n_estimators=5000, random_state=0, metric="multiclass"),
                param_grid=params, min_features_to_select=10, step=1,
                greater_is_better=False

            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=0,
                eval_metric='logloss'
            )
            print(model.estimator_)
            print(model.n_features_)
            print(model.ranking_)

            """
            lgbm_train = lgbm.Dataset(X_train, y_train)  # ,weight = train_weights)
            lgbm_valid = lgbm.Dataset(X_val, y_val, reference=lgbm_train)  # weight = val_weights)
            evaluation_results = {}
            model = lgbm.LGBMClassifier()
            
            rfe = RFE(model, n_features_to_select=args.n_features, verbose=2, step=args.step)

            print('Fitting model using RFE')
            rfe.fit(X_train, y_train)

            features_df = pd.DataFrame({'cols': X_train.columns, 'feat_rank': rfe.ranking_})
            features_df = features_df.sort_values(by='feat_rank', ascending=True)
            features_df.to_csv(f'{op_folder}features_ranked_fold{fold}.csv', index=False)

            # Get top n features. These will have rank 1
            features_df = features_df[features_df['feat_rank'] == 1]
            feature_cols = features_df["cols"].values
            with open(f'{op_folder}features_fold{fold}.pkl', 'wb') as f:
                pickle.dump(feature_cols, f)

        elif args.fs == 'eli5':
            model = lgbm.LGBMRegressor(**param_grid,
                                       device_type='cpu',
                                       n_estimators=100000)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric=rfe_auc,
                      # sample_weight=train_weights,
                      # eval_sample_weight=[val_weights],
                      verbose=500,
                      # categorical_feature = ['stock_id']
                      )

            perm = PermutationImportance(model, random_state=42, n_iter=args.n_trials)
            print('Fitting PermImportance')
            perm.fit(X_val, y_val, cv=3)

            eliDf = pd.DataFrame()
            eliDf['fimp'] = perm.feature_importances_
            eliDf['f'] = features
            eliDf.sort_values(by='fimp', ascending=True).to_csv(f'{op_folder}eliDf_fold_' + str(fold) + '.csv')


        else:
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                "num_class": 3,
                'metric': 'multiclass',  # multi_logloss #multiclass
                'learning_rate': 0.008,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'verbosity': 1,
                "max_depth": -1,
                "num_leaves": 15,
                "max_bin": 32,
                'is_unbalance': 'true'
            }
            lgbm_train = lgbm.Dataset(X_train, y_train)  # ,weight = train_weights)
            lgbm_valid = lgbm.Dataset(X_val, y_val, reference=lgbm_train)  # weight = val_weights)
            evaluation_results = {}
            model = lgbm.train(params=params,
                               train_set=lgbm_train,
                               valid_sets=[lgbm_train, lgbm_valid],
                               evals_result=evaluation_results,
                               verbose_eval=300,
                               num_boost_round=5000,
                               early_stopping_rounds=300,
                               categorical_feature=['discourse_type'],
                               )


# define your own objective function, make sure the function receives four parameters,
#  fit your model and return the objective value !
def objective_function_topass(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    P = rmspe(y_valid, model.predict(X_valid))
    return P


class RMSPE(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return rmspe(y_true, y_score)


class RMSE(Metric):
    def __init__(self):
        self._name = "rmse"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return rmse(y_true, y_score)


def ga(data):
    """
    Genetic Algorithm
    :param data:
    :return:
    """
    for fold in range(5):
        print(f'Fold {fold}')

        seed = 29
        param_grid = {
            'learning_rate': 0.1,
            'lambda_l1': 2,
            'lambda_l2': 7,
            'num_leaves': 1000,
            'min_sum_hessian_in_leaf': 20,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.6,
            'bagging_fraction': 0.6,
            'bagging_freq': 42,
            'min_data_in_leaf': 700,
            'max_depth': 5,
            # 'early_stopping_rounds': 300,
            'seed': seed,
            'feature_fraction_seed': seed,
            'bagging_seed': seed,
            'drop_seed': seed,
            'data_random_seed': seed,
            'objective': 'rmse',
            'boosting': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'device': 'gpu'
        }

        train_data = data[data['fold'] != fold].reset_index(drop=True)
        valid_data = data[data['fold'] == fold].reset_index(drop=True)

        train_idx = data[data['fold'] != fold].index.values
        valid_idx = data[data['fold'] == fold].index.values

        y_train = train_data[TARGET_COL].values
        y_val = valid_data[TARGET_COL].values

        features = list(train_data.columns)
        features.remove(TARGET_COL)
        features.remove(ID_COL)

        # create object of algorithm
        algo_object = GeneticOptimization(objective_function_topass, n_iteration=args.n_trials,
                                          population_size=len(features), selective_pressure=2, elitism=10,
                                          mutation_rate=0.07, minimize=True)

        X_train = train_data[features]
        X_val = valid_data[features]

        model = lgbm.LGBMRegressor(**param_grid)
        # remove args.step% of features each step

        ga = algo_object.fit(model, X_train, y_train, X_val, y_val, verbose=True)
        features = ga.best_feature_list
        print(features)
        with open(f'{op_folder}features_fold{fold}.pkl', 'wb') as f:
            pickle.dump(features, f)


# Optuna objective for hyperparams search
def objective(trial: Trial, data) -> float:
    global params
    seed = 42
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
        'lambda_l1': trial.suggest_int('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_int('lambda_l2', 0, 5),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.1, 1, 0.01),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 42),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'max_bin': trial.suggest_int('max_bin', 10, 100),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'n_estimators': trial.suggest_int('n_estimators', 5000, 100000),
        'early_stopping_rounds': 300,
        'seed': seed,
        'feature_fraction_seed':seed,
        'bagging_seed': seed,
        'drop_seed': seed,
        'data_random_seed': seed,
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multiclass',
        'boosting': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'device': 'cpu',
        'is_unbalance': 'true',
    }

    for fold in range(8):
        if args.run_folds and fold not in args.run_folds:
            print(f'Skipping fold {fold}')
            continue
        val_score, _ = train_folds(data, fold)
        return val_score


def rmspe(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def rmse(y_true, y_pred):
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    return np.sqrt(mse)


def rmspe_torch(y_true, y_pred):
    return (torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true))))


def feval_RMSPE(preds, lgbm_train):
    labels = lgbm_train.get_label()
    return 'RMSPE', round(rmspe(y_true=labels, y_pred=preds), 5), False

def feval_logloss(preds, lgbm_train):
    labels = lgbm_train.get_label()
    preds = [np.argmax(line) for line in preds]
    return 'LOGLOSS', round(log_loss(y_true=labels, y_pred=preds, labels=[0,1,2]), 5), False

def feval_RMSE(preds, lgbm_train):
    labels = lgbm_train.get_label()
    return 'RMSE', round(rmse(y_true=labels, y_pred=preds), 5), False


def feval_auc(preds, lgbm_train):
    labels = lgbm_train.get_label()
    return 'AUC', round(roc_auc_score(labels, preds), 5), True


import scipy as sp


def get_score(y_true, y_pred):
    #print(f'y_true: {y_true[0:5]} y_pred: {y_pred[0:5]}')
    #print(set(y_true), set(y_pred))
    score = log_loss(y_true, y_pred, labels=[0,1,2])
    return score


def feval_xgb_RMSPE(preds, xgb_train):
    labels = xgb_train.get_label()
    return 'RMSPE', round(rmspe(y_true=labels, y_pred=preds), 5)


def feval_xgb_RMSE(preds, xgb_train):
    labels = xgb_train.get_label()
    return 'RMSE', round(rmse(y_true=labels, y_pred=preds), 5)


def rfe_RMSPE(preds, labels):
    return 'RMSPE', round(rmspe(y_true=labels, y_pred=preds), 5), False


def rfe_auc(preds, labels):
    return 'AUC', round(roc_auc_score(labels, preds), 5), True


def infer(df, is_submission=True):
    if is_submission:
        test_df = pd.read_csv('../input/test.csv')
        print('Getting test from preprocessor')
        data = test_df
    else:
        data = df

    print('Doing Feature enginerring on data')
    print(f'Previous data shape: {data.shape}')
    features = list(data)
    gc.collect()
    start_fold = 0
    pred_df = pd.DataFrame()
    X_test = data[features]
    print(X_test.head())
    print(f'Final data shape {X_test.shape}')
    # X_test['stock_id'] = X_test['stock_id'].astype(int)

    for fold in range(5):
        print(f'Running inference on fold {fold}')
        model = lgbm.Booster(model_file=f'{args.infer_dir}/gbm_model_fold{fold}.lgb')
        pred_df[f'fold{start_fold}'] = model.predict(X_test)
        start_fold += 1

    print(pred_df.head())

    predictions = pred_df.mean(axis=1)
    if is_submission:
        print(predictions)
    else:
        val_score = rmspe(data[TARGET_COL].values, predictions)

        print(f'Val score: {val_score}')

#credit Olivier =) https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encoder(trn_series=None,
                   tst_series=None,
                   target=None,
                   min_samples_leaf=1,
                   smoothing=1,
                   noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name, how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def target_encode(train, test, target, column_list, drop_original=True):
    for var in column_list:
        print('Target encoding', var)
        trn_enc, test_enc = target_encoder(train[var],
                                           test[var],
                                           target=train[target],
                                           min_samples_leaf=100,
                                           smoothing=10,
                                           noise_level=0.01)

        train[var + '_TE'] = trn_enc
        test[var + '_TE'] = test_enc

    # Drop original columns
    if drop_original:
        train.drop(columns=column_list, inplace=True)
        test.drop(columns=column_list, inplace=True)

    return train, test

def get_svd(df, svd_model=None):
    x = df[['model_0_Effective', 'model_0_Ineffective', 'model_0_Adequate',
            'model_1_Effective', 'model_1_Ineffective', 'model_1_Adequate',
            'model_2_Effective', 'model_2_Ineffective', 'model_2_Adequate',
            'model_3_Effective', 'model_3_Ineffective', 'model_3_Adequate',
            'model_4_Effective', 'model_4_Ineffective', 'model_4_Adequate',
            'model_5_Effective', 'model_5_Ineffective', 'model_5_Adequate',
            'model_6_Effective', 'model_6_Ineffective', 'model_6_Adequate',
            'model_7_Effective', 'model_7_Ineffective', 'model_7_Adequate',
            ]].values
    if svd_model:
        x_svd = svd_model.transform(x)
    else:
        svd = TruncatedSVD(n_components=2)
        x_svd = svd.fit_transform(x)

    df.loc[:,['svd0', 'svd1']] = x_svd
    return df, svd

# Main train loop
def train_folds(data, fold):
    global params
    print(f'Final params:{params}')
    with open(op_folder + 'params.txt', 'a+') as fid:
        fid.write(f'\nparams = {params}')
    if args.algo == 'tabnet':
        data = data.fillna(data.mean())
        scaler = StandardScaler()
        scaled_cols = data.drop(['fold', TARGET_COL], axis=1).columns.to_list()
        scaler.fit(data[scaled_cols])
        with open(f'{op_folder}scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)



    #data.dropna(inplace=True)
    #print(data['fold'].value_counts())
    train_data = data[data['kfold'] != fold].reset_index(drop=True)
    valid_data = data[data['kfold'] == fold].reset_index(drop=True)

    if args.add_features:
        print('Doing Feature engineering on data')
        print(f'Previous data shape: {train_data.shape}')
        """
        print(f'Adding TE features')
        train_data, valid_data = target_encode(train_data, valid_data, TARGET_COL, ['freq_of_essay_id', 'num_essay_words', 'topic'], drop_original=False)
        """

        print(f'Adding SVD features')
        train_data, svd_model = get_svd(train_data)
        with open(f'{op_folder}svd_model_fold{fold}.pkl', 'wb') as f:
            pickle.dump(svd_model, f)
        valid_data, _ = get_svd(valid_data)


        gc.collect()
        print(f'After FE data shape: {train_data.shape}')


    print(train_data.head())

    #print(valid_data.head())
    oof_df = pd.DataFrame()
    oof_df[ID_COL] = valid_data[ID_COL]

    # Remove columns from dataframe
    if not args.features:
        train_data = train_data.drop(drop_columns, axis=1)
        valid_data = valid_data.drop(drop_columns, axis=1)


    y_train = train_data[TARGET_COL].values
    y_val = valid_data[TARGET_COL].values

    features = list(train_data.columns)
    features.remove(TARGET_COL)
    # features.remove(ID_COL)
    features.remove('kfold')
    # print(f'Features: {features}')

    if args.algo == 'tabnet':
        # features.remove('stock_id')
        # Scaler transform data
        X_train = scaler.transform(train_data[scaled_cols])
        X_train = pd.DataFrame(X_train)
        X_val = scaler.transform(valid_data[scaled_cols])
        X_val = pd.DataFrame(X_val)
        # Add label encoded stock id as last column
        # X_train['stock_id'] = train_data['stock_id']
        # X_val['stock_id'] = valid_data['stock_id']
        # Back to arrays
        X_train = X_train.values
        X_val = X_val.values

    else:
        X_train = train_data[features]
        X_val = valid_data[features]

    if args.features:
        print(f'User speficied {args.features}. Using that instead of all features')
        with open(args.features, 'rb') as f:
            features = pickle.load(f)
        X_train = X_train[features]
        X_val = X_val[features]

    # Dump features so that we can refer to this easily in inference
    if not args.kmeans_agg:
        with open(f'{op_folder}features.pkl', 'wb') as f:
            pickle.dump(features, f)

    # train_weights = 1/np.square(y_train)
    # val_weights = 1/np.square(y_val)

    if args.svd or args.pca:
        if args.svd:
            print(f'User specified SVD with n_components: {args.svd}')
            dim_red = TruncatedSVD(n_components=args.svd, n_iter=7, random_state=42)

        if args.pca:
            print(f'User specified PCA with n_components: {args.pca}')
            dim_red = PCA(n_components=args.pca)

        # X_train  = X_train.fillna(X_train.mean())
        # X_val = X_val.fillna(X_train.mean())

        scaler = MinMaxScaler()

        svd_features = features

        X_train_data = scaler.fit_transform(X_train[svd_features])
        X_val_data = scaler.transform(X_val[svd_features])

        X_train_svd = dim_red.fit_transform(X_train_data)
        X_val_svd = dim_red.transform(X_val_data)
        X_train_pre = pd.DataFrame(X_train_svd)  # , columns=[i for i in range(args.svd)])
        X_val_pre = pd.DataFrame(X_val_svd)  # , columns=[i for i in range(args.svd)])

        X_train = X_train_pre
        X_val = X_val_pre

    print(f'Feature shape: {X_train.shape}')
    print(f'Fold: {fold}\nTrain: {X_train.shape}\nVal:{X_val.shape}')

    if args.algo == 'lgb':
        if args.use_autolgb:
            # Use auto lgb on first fold to get hyper-params
            if fold == 0:
                clf = AutoLGB(objective='regression', metric='rmse', random_state=1234)
                clf.tune(pd.DataFrame(X_train), pd.DataFrame(y_train))
                params = clf.params
                print(f'{params}')
                print(f'Features: {clf.features}')

        lgbm_train = lgbm.Dataset(X_train, y_train)  # ,weight = train_weights)
        lgbm_valid = lgbm.Dataset(X_val, y_val, reference=lgbm_train)  # weight = val_weights)
        evaluation_results = {}
        model = lgbm.train(params=params,
                           train_set=lgbm_train,
                           valid_sets=[lgbm_train, lgbm_valid],
                           evals_result=evaluation_results,
                           verbose_eval=300,
                           num_boost_round=5000,
                           early_stopping_rounds=300,
                           categorical_feature=['discourse_type'],
                           )

        val_pred = model.predict(X_val)
    elif args.algo == 'xgb':
        dtrain = xgb.DMatrix(X_train, label=y_train)  # , weight= train_weights)
        dval = xgb.DMatrix(X_val, label=y_val)  # , weight = val_weights)

        model = xgb.train(params=params,
                          dtrain=dtrain,
                          num_boost_round=10000,
                          early_stopping_rounds=100,
                          feval=feval_xgb_RMSE,
                          verbose_eval=100,
                          evals=[(dtrain, 'dtrain'), (dval, 'dval')],
                          )
        val_pred = model.predict(dval)
    elif args.algo == 'svm':
        # Scale feature
        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.transform(X_train)
        model = SVC(C=10, kernel='rbf', gamma='auto', probability=True)
        model.fit(X_train_std, y_train)
        X_val_std = scaler.transform(X_val)
        val_pred = model.predict_proba(X_val_std)
    elif args.algo == 'ridge':
        model = Ridge(alpha=40.0)
        # scaler = StandardScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        model.fit(X_train, y_train)
        # X_val =scaler.transform(X_val)
        val_pred = model.predict_proba(X_val)
    elif args.algo == 'rf':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)
    elif args.algo == 'tabnet':
        y_train = y_train.reshape(-1, 1)
        y_val_ = y_val.reshape(-1, 1)

        model = TabNetRegressor(**params)
        print('Fitting tabnet')
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val_)],
            eval_name=['train', 'valid'],
            eval_metric=[RMSE],
            max_epochs=args.epochs,
            patience=10,
            batch_size=1024 * 2, virtual_batch_size=128 * 2,
            num_workers=4,
            drop_last=False,
            # weights = train_weights,
            pin_memory=False,
            loss_fn=nn.L1Loss(),
        )

        val_pred = model.predict(X_val)

    if args.algo != 'fastai_tabular':
        # val_score = roc_auc_score(y_val, val_pred) #rmspe(y_val, val_pred)
        val_score = get_score(y_val, val_pred)

    # Dump files to model dir
    with open(op_folder + 'params.txt', 'a+') as fid:
        fid.write(f'\nFold = {fold} Val Score= {val_score}')

    oof_df['pred'] = [np.argmax(line) for line in val_pred]

    print('\nFold = {} Val Score= {}'.format(fold, val_score))
    print('Saving model..')
    if args.algo == 'lgb':
        model.save_model(op_folder + 'gbm_model_fold' + str(fold) + f'.lgb')

        # Feature importance
        train_columns = list(X_train.columns)
        varimp_df = pd.DataFrame([train_columns, model.feature_importance()]).T
        varimp_df.columns = ['feature', 'imp']
        varimp_df = varimp_df.sort_values(by='imp', ascending=False)
        varimp_df.to_csv(f'{op_folder}feat_imp_fold{fold}.csv', index=False)

    elif args.algo == 'xgb':
        model.save_model(op_folder + 'gbm_model_fold' + str(fold) + f'.xgb')

        # Feature importance
        """
        train_columns = list(X_train.columns)
        varimp_df = pd.DataFrame([train_columns, model.feature_importance_]).T
        varimp_df.columns = ['feature', 'imp']
        varimp_df = varimp_df.sort_values(by='imp', ascending=False)
        varimp_df.to_csv(f'{op_folder}feat_imp_fold{fold}.csv', index=False)
        """
    elif args.algo == 'svm':
        dump(model, open(f'{op_folder}svr_model_fold{fold}.pkl', 'wb'))
        dump(scaler, open(f'{op_folder}svr_scaler_fold{fold}.pkl', 'wb'))
    elif args.algo == 'ridge':
        dump(model, open(f'{op_folder}ridge_model_fold{fold}.pkl', 'wb'))
        # dump(scaler, open(f'{op_folder}ridge_scaler_fold{fold}.pkl', 'wb'))
    elif args.algo == 'rf':
        dump(model, open(f'{op_folder}rf_model_fold{fold}.pkl', 'wb'))
    elif args.algo == 'tabnet':
        save_model_path = f'{op_folder}tabnet_model_fold{fold}'
        model.save_model(save_model_path)
    elif args.algo == 'nn':
        save_model_path = f'{op_folder}keras_model_fold{fold}.h5'
        model.save(save_model_path, save_format='h5')

    gc.collect()

    return val_score, oof_df


if not args.debug:
    df = pd.read_csv(args.train_file)
else:
    df = pd.read_csv(args.train_file, nrows=args.debug)

le = LabelEncoder()
"""
if args.algo == 'tabnet':
    le.fit(df["stock_id"])
    df["stock_id"] = le.transform(df["stock_id"])
    with open(f'{op_folder}le.pkl', 'wb') as f:
        pickle.dump(le, f)
"""
df['kfold'] = df['kfold'].astype(int)
target_dict = {"Ineffective": 0, "Adequate": 1, "Effective": 2}
dt_dict = {"Lead": 0, "Claim": 1, "Evidence": 2, "Counterclaim": 3, "Rebuttal": 4,
           "Concluding Statement": 5, "Position": 6
           }


df['discourse_effectiveness'] = df['discourse_effectiveness'].map(target_dict)
df['discourse_type'] = df['discourse_type'].map(dt_dict)
#df['pred'] = df['pred'].map(target_dict)

# set params
if args.algo == 'xgb':
    params = {
        'learning_rate': 0.1,
        'colsample_bytree': 0.8,
        'colsample_bynode': 0.8,
        'max_depth': 5,
        'subsample': 0.9,
        'objective': 'reg:squarederror',
        'eval_metric': 'binary',
        'tree_method': 'gpu_hist',
        'n_jobs': -1,
        'seed': 42,
        'reg_alpha': 2,
        'reg_lambda': 10,
    }

elif args.algo == 'lgb':
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        "num_class": 3,
        'metric': 'multiclass', #multi_logloss #multiclass
        'learning_rate': 0.008,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'verbosity': 1,
        "max_depth": -1,
        "num_leaves": 15,
        "max_bin": 32,
        'is_unbalance': 'true'
    }

    if args.svd or args.pca:
        params = {
            'learning_rate': 0.01,
            'lambda_l1': 2,
            'lambda_l2': 7,
            'num_leaves': 2000,
            'min_sum_hessian_in_leaf': 30,
            'feature_fraction': 0.7,
            'feature_fraction_bynode': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 42,
            'min_data_in_leaf': 1000,
            'max_depth': -1,
            # 'max_bin': 100,
            'early_stopping_rounds': 300,
            'seed': seed,
            'feature_fraction_seed': seed,
            'bagging_seed': seed,
            'drop_seed': seed,
            'data_random_seed': seed,
            'objective': 'rmse',
            'boosting': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'device': 'cpu',
            'n_estimators': 100000
        }
elif args.algo == 'tabnet':
    params = dict(
        n_d=64,
        n_a=64,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=0,
        optimizer_fn=optim.Adam,
        optimizer_params=dict(lr=args.lr, weight_decay=1e-5),
        mask_type="entmax",
        scheduler_params=dict(
            mode="min", patience=5, min_lr=1e-5, factor=0.9),
        scheduler_fn=ReduceLROnPlateau,
        seed=42,
        # verbose = 5,
        # cat_dims=[len(le.classes_)], cat_emb_dim=[10], cat_idxs=[-1] # define categorical features
    )
else:
    params = None

# Dump files to model dir
if args.algo != 'tabnet':
    if args.optuna:
        open_with = 'a+'
    else:
        open_with = 'w'

    with open(op_folder + 'params.txt', f'{open_with}') as fid:
        fid.write(json.dumps(params))
        fid.write('Arguments:\n')
        attrs = vars(args)
        fid.write(', '.join("%s: %s" % item for item in attrs.items()))
        fid.write('\n\n')



# Hyperparam search using optna
if args.optuna:
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, df), n_trials=args.n_trials)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))
    with open(op_folder + 'params.txt', 'a+') as fid:
        fid.write('\n')
        fid.write('----'*5)
        fid.write('\nBest trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))
        fid.write('\n')
        fid.write('----'*5)
    seed = 42
    base_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        "num_class": 3,
        'metric': 'multiclass',
        'learning_rate': 0.008,
        #'colsample_bytree': 0.8,
        #'subsample': 0.8,
        #'verbosity': 1,
        #"max_depth": 6,
        #"num_leaves": 15,
        #"max_bin": 32,
        'is_unbalance': 'true',
        'seed': seed,
        'feature_fraction_seed':seed,
        'bagging_seed': seed,
        'drop_seed': seed,
        'data_random_seed': seed,
    }
    params = base_params
    params.update(study.best_trial.params)
    for fold in range(8):
        print(f'Fold: {fold}')
        if args.run_folds and fold not in args.run_folds:
            print(f'Skipping fold {fold}')
            continue
        val_score, _ = train_folds(df, fold)

elif args.fs:
    feature_selection(df)
elif args.ga:
    ga(df)
elif args.full_data:
    pass
elif args.infer:
    infer(df, is_submission=args.submit)
else:
    val_scores = []
    oof_df = pd.DataFrame()
    for fold in range(8):
        if args.run_folds and fold not in args.run_folds:
            print(f'Skipping fold {fold}')
            continue
        val_score, oof_df_ = train_folds(df, fold)
        val_scores.append(val_score)
        oof_df = pd.concat([oof_df, oof_df_])
    #oof_df.to_csv('oof_df.csv', index=False)
    print(oof_df.shape)
    # Take mean of all val scores
    val_score = np.mean(val_scores)
    print(f'Mean CV: {val_score}')
    # add val score to params.txt
    with open(op_folder + 'params.txt', 'a') as fid:
        fid.write(f'Mean CV: {val_score}')


