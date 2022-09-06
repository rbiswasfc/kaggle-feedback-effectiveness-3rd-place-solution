import os
import gc
import glob
import json
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, log_loss
import matplotlib.pyplot as plt
from itertools import chain

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import lightgbm as lgbm

###################################### functions for creating features #########################

def clean_text(x):
    x = re.sub("[^-9A-Za-z ]", "", x).lower()

    # removing stopwords and tokenizing

    stop = stopwords.words("english")

    tokens = [word for word in (token for token in word_tokenize(x)) if word not in stop]

    # lemmatizing

    lmtzr = nltk.WordNetLemmatizer()

    preprocessed_text = ' '.join([lmtzr.lemmatize(word) for word in tokens])

    return preprocessed_text

def position_of_discourse(x):
    lis = x['essay_text'].split(x['discourse_text'])
    if len(lis) == 1:
        return 1
    else:
        num_sent_before = sent_tokenize(lis[0])
        num_sent_after = (x['discourse_text']).join(lis[1:])
        num_sent_after = sent_tokenize(num_sent_after)
        return (len(num_sent_before)/(len(num_sent_after)+1))

def get_substring_span(text, substring, min_length=10, fraction=0.999):
    """
    Returns substring's span from the given text with the certain precision.
    """

    position = text.find(substring)
    substring_length = len(substring)
    if position == -1:
        half_length = int(substring_length * fraction)
        half_substring = substring[:half_length]
        half_substring_length = len(half_substring)
        if half_substring_length < min_length:
            return [-1, 0]
        else:
            return get_substring_span(text=text,
                                      substring=half_substring,
                                      min_length=min_length,
                                      fraction=fraction)

    span = [position, position+substring_length]
    return span

# functions for separating the POS Tags
def adjectives(text):
    blob = TextBlob(text)
    return len([word for (word,tag) in blob.tags if tag == 'JJ'])
def verbs(text):
    blob = TextBlob(text)
    return len([word for (word,tag) in blob.tags if tag.startswith('VB')])
def adverbs(text):
    blob = TextBlob(text)
    return len([word for (word,tag) in blob.tags if tag.startswith('RB')])
def nouns(text):
    blob = TextBlob(text)
    return len([word for (word,tag) in blob.tags if tag.startswith('NN')])

def return_typ_tokens(text):
    blob = TextBlob(text)
    sum_all = len(blob.tags)
    unique_typ = len(set([tag for (word,tag) in blob.tags]))
    return unique_typ/sum_all

def tags(text):
    blob = TextBlob(text)
    return blob

def count_typ_tags(text, typ):
    return len([word for (word,tag) in text.tags if tag.startswith(typ)])

discourse_type2id = {
    "Lead": 1,
    "Position": 2,
    "Claim": 3,
    "Counterclaim": 4,
    "Rebuttal": 5,
    "Evidence": 6,
    "Concluding Statement": 7,
}

############################################### Modeling ##########################################
class DataBunch:
    """
    data container
    """
    def __init__(self, features, cat_features,X_train, y_train, X_valid=None, y_valid=None):
        self.features = features
        self.cat_features = cat_features
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

class LGBModel(object):
    def __init__(self, params, fold):
        self.params = params
        self.model_output_dir = config["model_dir"]
        self.fold = fold

    def train(self, data, target='label', n_trees=5000, esr=500):
        features = data.features
        cat_features = sorted(data.cat_features)
        self.target = target
        self.features = features

        X_train, X_valid = data.X_train, data.X_valid
        y_train, y_valid = data.y_train, data.y_valid

        train_data = lgbm.Dataset(X_train[features], 
                                  label=y_train, 
                                  feature_name=features)
        
        valid_data = lgbm.Dataset(X_valid[features], 
                                  label=y_valid, 
                                  feature_name=features, 
                                  reference=train_data)
        evaluation_results = {}
        
        self.model = lgbm.train(
            self.params,
            train_data,
            num_boost_round=n_trees,
            valid_sets=[train_data, valid_data],
            evals_result=evaluation_results,
            early_stopping_rounds=esr,
            categorical_feature=cat_features,
            verbose_eval=250
        )
        
        print("Training Done ...")
        return evaluation_results

    def predict(self, df):
        preds = self.model.predict(df[self.features], 
                                   num_iteration=self.model.best_iteration)
        return preds

    def get_feature_importance(self):
        df_imp = pd.DataFrame({'imp': self.model.feature_importance(importance_type='gain'),
                               'feature_name': self.model.feature_name()})
        df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)
        return df_imp

    def save_model(self, fold):
        model = self.model
        save_path = os.path.join(self.model_dir, "lgbm_model_fold_{}.txt".format(fold))
        model.save_model(save_path, num_iteration=model.best_iteration)

    def load_model(self, fold):
        save_path = os.path.join(self.model_dir, "lgbm_model_fold_{}.txt".format(fold))
        model = lgbm.Booster(model_file=save_path)
        return model

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    "num_class" : 3,
    'metric': 'multiclass',
    'learning_rate': 0.008,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': 1,
    "max_depth": 6,
    "num_leaves": 15,
    "max_bin": 32,
    'is_unbalance': 'true'
}

########################################## Training per fold ########################################

def run_training():
    #-------- seed ------------#
    print("=="*40)
    args = parse_args()
    config = read_config(args)

    if config["use_random_seed"]:
        seed = random.randint(401, 999)
        config["seed"] = seed

    # seed = random.randint(401, 999)
    seed = config["seed"]
    print(f"setting seed: {seed}")
    seed_everything(seed)

    os.makedirs(config["model_dir"], exist_ok=True)
    oof_dfs = []
    for csv in config["oof_csvs"]:
        oof_df = pd.read_csv(csv)
        oof_dfs.append(oof_df)

    pred_cols = ["Ineffective", "Adequate", "Effective"]
    for model_idx in range(len(oof_dfs)):
        col_map = dict()
        for col in pred_cols:
            col_map[col] = f"model_{model_idx}_{col}"
        oof_dfs[model_idx] = oof_dfs[model_idx].rename(columns=col_map)

    merged_df = oof_dfs[0]

    for oof_df in oof_dfs[1:]:
        keep_cols = ["discourse_id"] + [col for col in oof_df.columns if col.startswith("model")]
        oof_df = oof_df[keep_cols].copy()
        merged_df = pd.merge(merged_df, oof_df, on="discourse_id", how='inner')
    assert merged_df.shape[0] == oof_dfs[0].shape[0]

    feature_names = [col for col in merged_df.columns if col.startswith("model")]

    config["cat_features"] = []
    config["features"] = feature_names

    ################################ read data ###########################

    df = pd.read_csv(os.path.join(config["fpe_dataset_dir"], "train.csv"))
    fold_df = pd.read_parquet(config["fold_path"])
    df = pd.merge(df, fold_df, on="essay_id", how="left")
    essay_df = pd.read_parquet(config["train_essay_fpe22_dir"])
    meta_df = pd.merge(merged_df, df, on="discourse_id", how="left")
    df = pd.merge(meta_df, essay_df, on="essay_id", how="left")

    ################################ feature engg ############################

    df = df.sort_values(by = ['essay_id','discourse_id']).reset_index(drop = True)
    print('Processing spans & feature engg')
    df["discourse_span"] = df[["essay_text", "discourse_text"]].apply(lambda x: get_substring_span(x[0], x[1]), axis=1)
    df["discourse_start"] = df["discourse_span"].apply(lambda x: x[0])
    df["discourse_end"] = df["discourse_span"].apply(lambda x: x[1])

    df['discourse_len'] = df['discourse_end'] - df['discourse_start']
    df['freq_of_essay_id'] = df['essay_id'].map(dict(df['essay_id'].value_counts()))

    df["discourse_type"] = df["discourse_type"].map(discourse_type2id)
    df['discourse_type_fe'] = df['discourse_type'].map(dict(df['discourse_type'].value_counts()))

    df['blob_discourse'] = df['discourse_text'].apply(tags)
    df['discourse_Adjectives'] = df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'JJ'))
    df['discourse_Verbs'] = df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'VB'))
    df['discourse_Adverbs'] = df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'RB'))
    df['discourse_Nouns'] = df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'NN'))
    df['discourse_VBP'] = df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'VBP'))
    df['discourse_PRP'] = df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'PRP'))
    df['count_next_line_essay'] = df['essay_text'].apply(lambda x: x.count("\n\n"))

    essay_discourse = df.groupby(['essay_id']).apply(lambda x: \
                            x['discourse_type'].nunique()).reset_index()
    essay_discourse.rename(columns = {0:'unique_discourse_type'}, inplace = True)

    df = df.merge(essay_discourse[['essay_id','unique_discourse_type']], on = 'essay_id',\
              how = 'left')

    new_col = []
    for unique in ['Claim']:
        df['is_' + unique] = df['discourse_type'].apply(lambda x: 1 if x == unique else 0)
        new_col.append('is_'+unique)

    ############################## updating features ################################

    config["features"].extend(["discourse_type",
                "discourse_type_fe","discourse_len","freq_of_essay_id",\
                          "unique_discourse_type"]+new_col)
    config['features'].extend(['discourse_Adjectives','discourse_Verbs',\
                          'discourse_Adverbs','discourse_Nouns',\
                          'count_next_line_essay','discourse_VBP','discourse_PRP'])
    config["cat_features"].append("discourse_type")

    fold = args.fold
    config["train_folds"] = [i for i in range(config["n_folds"]) if i != fold]
    config["valid_folds"] = [fold]
    config["fold"] = fold
    print(f"train folds: {config['train_folds']}")
    print(f"valid folds: {config['valid_folds']}")
    print("=="*40)

    config["num_features"] = len(feature_names)
    

    if config["debug"]:
        print("DEBUG Mode: sampling 1024 examples from train data")
        df = df.sample(min(1024, len(df)))


    # create the dataset
    print("creating the datasets and data loaders...")
    train_df = df[df["kfold"].isin(config["train_folds"])].copy()
    valid_df = df[df["kfold"].isin(config["valid_folds"])].copy()

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")

    
    # create the model and optimizer
    print("starting training")
    target = "label"
    X_train, y_train = train_df, train_df[target].values
    X_valid, y_valid = valid_df, valid_df[target].values
    
    X_train.reset_index(drop = True, inplace = True)
    X_valid.reset_index(drop = True, inplace = True)

    model = LGBModel(params, fold)
    data = DataBunch(config["features"], config["cat_features"], X_train, y_train, X_valid, y_valid)
    model.train(data, target)
    pred = model.predict(X_valid)
    valid_df[labels] = pred
    save_path = os.path.join(config["model_dir"], "oof_lgbm_model_fold_{}.csv".format(fold))
    valid_df.to_csv(save_path, index = False)

    model.save_model(fold)

if __name__ == "__main__":
    run_training()
