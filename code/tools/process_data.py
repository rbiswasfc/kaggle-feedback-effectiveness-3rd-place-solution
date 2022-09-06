import argparse
import os

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

topic_map = {
	'seagoing luke animals cowboys': 'Should you join the Seagoing Cowboys program?',
	'driving phone phones cell' :  'Should drivers be allowed to use cell phones while driving?',
	 'phones cell cell phones school': 'Should students be allowed to use cell phones in school?',
	 'straights state welfare wa' : ' State welfare' ,
	 'summer students project projects': 'Should school summer projects be designed by students or teachers?',
	 'students online school classes': 'Is distance learning or online schooling beneficial to students?',
	 'car cars usage pollution': 'Should car usage be limited to help reduce pollution?',
	 'cars driverless car driverless cars': 'Are driverless cars going to be helpful?',
	 'emotions technology facial computer' : 'Should computers read the emotional expressions of students in a classroom?',
	 'community service community service help': 'Should community service be mandatory for all students?',
	 'sports average school students' : 'Should students be allowed to participate in sports  unless they have at least a grade B average?',
	 'advice people ask multiple': 'Should you ask multiple people for advice?',
	 'extracurricular activities activity students': 'Should all students participate in at least one extracurricular activity?',
	 'electoral college electoral college vote':  'Should the electoral college be abolished in favor of popular vote?' ,
	 'electoral vote college electoral college' : 'Should the electoral college be abolished in favor of popular vote?' ,
	 'face mars landform aliens' : 'Is the face on Mars  a natural landform or made by Aliens?',
     'venus planet author earth': 'Is Studying Venus a worthy pursuit?',
}

def create_cv_folds(df, n_splits=5):
    """create cross validation folds

    :param df: input dataframe of labelled data
    :type df: pd.DataFrame
    :param n_splits: how many cross validation splits to perform, defaults to 5
    :type n_splits: int, optional
    :return: dataframe with kfold column added
    :rtype: pd.DataFrame
    """
    kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=1300)
    for fold, (trn_, val_) in enumerate(kfold.split(df.discourse_id, df.effective_topic, df.essay_id)):
        print(len(trn_), len(val_))
        df.loc[val_, "kfold"] = fold
    return df


def _load_essay(essay_id, is_train, data_dir):
    if is_train:
        filename = os.path.join(data_dir, "train", essay_id + ".txt")
    else:
        filename = os.path.join(data_dir, "test", essay_id + ".txt")

    with open(filename, "r") as f:
        text = f.read()

    return [essay_id, text]


def read_essays(essay_ids, is_train=True, num_jobs=2, data_dir="../datasets/feedback-prize-effectiveness"):
    train_essays = []

    results = Parallel(n_jobs=num_jobs, verbose=1)(delayed(_load_essay)(essay_id, is_train, data_dir)
                                                   for essay_id in essay_ids)
    for result in results:
        train_essays.append(result)

    result_dict = dict()
    for e in train_essays:
        result_dict[e[0]] = e[1]

    essay_df = pd.Series(result_dict).reset_index()
    essay_df.columns = ["essay_id", "essay_text"]
    return essay_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    args = ap.parse_args()

    n_splits = [10]
    for n_split in n_splits:
        file_name = f"cv_map_topics_{n_split}_folds.parquet"
        train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        topics_df = pd.read_csv("topics.csv")
        train_df = train_df.merge(topics_df)
        train_df['topic'] = train_df['topic'].map(topic_map)
        train_df['effective_topic'] = train_df['discourse_effectiveness'] + " " + train_df['topic']
        print(train_df.head())
        train_df = create_cv_folds(train_df, n_splits=n_split)
        fold_df = train_df[["essay_id", "kfold"]].drop_duplicates()
        fold_df = fold_df.reset_index(drop=True)
        fold_df.to_parquet(os.path.join(args.save_dir, file_name))

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    train_essay_ids = train_df["essay_id"].unique().tolist()
    train_essay_df = read_essays(train_essay_ids, is_train=True, num_jobs=25,
                                 data_dir=args.data_dir)
    file_name = "fpe_22_train_essays.parquet"
    train_essay_df.to_parquet(os.path.join(args.save_dir, file_name))


if __name__ == "__main__":
    main()
