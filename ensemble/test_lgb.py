# %% [markdown]
# # LGB
import pandas as pd
import lightgbm as lgbm
import argparse

parser = argparse.ArgumentParser(description='Process pytorch params.')

parser.add_argument('-train_file', help='Algorithm to use')
parser.add_argument('-model_dir', help='Algorithm to use')


args = parser.parse_args()

print(args)

test_df = pd.read_csv(args.train_file)

debug = False
if debug:
    test_df = test_df.iloc[0:500]


# %% [code]
test_df.head()

dt_dict = {"Lead": 0, "Claim": 1, "Evidence": 2, "Counterclaim": 3, "Rebuttal": 4,
           "Concluding Statement": 5, "Position": 6
           }

test_df["discourse_type"] = test_df["discourse_type"].map(dt_dict)

# %% [code]
config = dict()
feature_names = [col for col in test_df.columns if col.startswith("model")]

config["features"] = feature_names


origial_fts = config["features"]
new_fts = ["discourse_type"] + origial_fts
config["features"] = new_fts

model_paths = [
    f"{args.model_dir}/gbm_model_fold0.lgb",
    f"{args.model_dir}/gbm_model_fold1.lgb",
    f"{args.model_dir}/gbm_model_fold2.lgb",
    f"{args.model_dir}/gbm_model_fold3.lgb",
    f"{args.model_dir}/gbm_model_fold4.lgb",
    f"{args.model_dir}/gbm_model_fold5.lgb",
    f"{args.model_dir}/gbm_model_fold6.lgb",
    f"{args.model_dir}/gbm_model_fold7.lgb",
]
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
        svd_model = TruncatedSVD(n_components=2)
        x_svd = svd.fit_transform(x)

    df.loc[:,['svd0', 'svd1']] = x_svd
    return df, svd_model


import pickle

config["features"].extend(['topic', 'num_discourse_words', 'num_discourse_chars', 'text_position', 'num_essay_words', 'num_essay_chars',
                           'freq_of_essay_id',
                           'discourse_Adjectives', 'discourse_Verbs', 'discourse_Adverbs', 'discourse_Nouns','count_next_line_essay'])

# %% [code]
all_df = pd.DataFrame()
for fold, mp in enumerate(model_paths):
    model = lgbm.Booster(model_file=mp)
    fold_df = test_df[test_df['kfold']==fold]
    #with open(f'{args.model_dir}/svd_model_fold{fold}.pkl', 'rb') as f:
     #   svd_model = pickle.load(f)
    #fold_df, _ = get_svd(fold_df, svd_model)
    print(f'Fold shape: {fold_df.shape}')
    #print(fold_df.head())
    preds = model.predict(fold_df[config["features"]], num_iteration=model.best_iteration)
    fold_df[["Ineffective", "Adequate", "Effective"]] = preds
    all_df = pd.concat([all_df, fold_df])
    print(f'All shape: {all_df.shape}')
    print('---'*5)


submission_df = all_df[['discourse_id', "Ineffective", "Adequate", "Effective"]]

# %% [code]
submission_df.to_csv("level2/lgb_ensemble.csv", index=False)