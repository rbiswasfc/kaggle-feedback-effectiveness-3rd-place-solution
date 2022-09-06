import ast
import os
import gc
import warnings
from nltk.corpus import stopwords
from textblob import TextBlob

all_stopwords = stopwords.words('english')
from tqdm import tqdm

tqdm.pandas()

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
import torch

print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import re
import textdistance as td
import swifter
from laserembeddings import Laser
from simcse import SimCSE

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# calculate tfidf and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse
from sklearn.decomposition import TruncatedSVD

parser = argparse.ArgumentParser(description='Process pytorch params.')

parser.add_argument('-train_file', help='Algorithm to use')
parser.add_argument('-out_file', help='Algorithm to use')
parser.add_argument('-add_embeddings', action='store_true', help='Calculate cosine similarity')
parser.add_argument('-embedding_type', type=str, nargs='+', default='td', choices=['lev', 'lcsstr', 'bert', 'tfidf', 'laser', 'simcse'], help='Debug mode')
parser.add_argument('-debug', type=int, help='Debug mode')
parser.add_argument('-add_bert', action='store_true', help='Add bert embeddings')
parser.add_argument('-add_fts', action='store_true')
parser.add_argument('-csvs', type=str, nargs='+', help='List of oof csvs', required=True)

args = parser.parse_args()

print(args)



laser = Laser()

def get_text_embeddings(train, val, column='discourse_text', max_features=4096):
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=max_features)  # max_df = 0.5
    train_embeddings = model.fit_transform(train[column]).toarray()
    val_embeddings = model.transform(val[column]).toarray()

    del model
    return train_embeddings, val_embeddings


def get_bert_embeddings(text):
    model_name = 'AI-Growth-Lab/PatentSBERTa'
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(text)
    return sentence_embeddings


def get_laser_embeddings(text):
    embeddings = laser.embed_sentences(text,lang='en')
    return embeddings

def np_cosine_similarity(u, v):
    u = np.expand_dims(u, 1)
    n = np.sum(u * v, axis=2)
    d = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=1)
    return n / d


embedding_dict = {}


# function to calculate tfidf and cosine similarity
def get_embeddings(row, embedding_type='bert'):
    text1 = row['discourse_text']
    text2 = row['essay_text']

    text_embeddings1 = None
    text_embeddings2 = None

    #print(f'Mode: {mode} text1: {text1} text2: {text2}')
    if embedding_type == 'tfidf':
        if text1 not in embedding_dict:
            text_embeddings1 = get_text_embeddings(text1)
            print(text1, text_embeddings1)
            embedding_dict[text1] = text_embeddings1
        else:
            text_embeddings1 = embedding_dict[text1]
        if text2 not in embedding_dict:
            text_embeddings2 = get_text_embeddings(text2)
            embedding_dict[text2] = text_embeddings2
        else:
            text_embeddings2 = embedding_dict[text2]



    elif embedding_type == 'bert':
        if mode == 'drop_context':
            bert_dict = np.load('bert_embeddings_anchor.npy', allow_pickle=True)
            if text1 not in bert_dict[()]:
                text_embeddings1 = get_bert_embeddings(text1).reshape(1, -1)
                bert_dict[()][text1] = text_embeddings1
            else:
                print(f'Context:{text1} not found')
                text_embeddings1 = bert_dict[()][text1]
        else:
            bert_dict = np.load('bert_embeddings.npy', allow_pickle=True)
            text_embeddings1 = bert_dict[()][text1]
            text_embeddings1 = text_embeddings1.reshape(1, -1)

        if mode == 'anchor':
            bert_text_dict = np.load('bert_embeddings_anchor.npy', allow_pickle=True)
        else:
            bert_text_dict = np.load('bert_embeddings_target.npy', allow_pickle=True)

        if text2 not in bert_text_dict[()]:
            text_embeddings2 = get_bert_embeddings(text2).reshape(1, -1)
            bert_text_dict[()][text2] = text_embeddings2
        else:
            text_embeddings2 = bert_text_dict[()][text2]
        similarity = np_cosine_similarity(text_embeddings1, text_embeddings2)
        similarity = similarity[0][0]
    elif embedding_type == 'laser':
        text_embeddings1 = get_laser_embeddings(text1)
        text_embeddings2 = get_laser_embeddings(text2)
    else:
        raise ValueError('Invalid embedding type')
    
    return text_embeddings1, text_embeddings2




import torch.nn.functional as F


# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T18:41:21.596626Z","iopub.execute_input":"2022-06-29T18:41:21.596956Z","iopub.status.idle":"2022-06-29T18:41:21.603473Z","shell.execute_reply.started":"2022-06-29T18:41:21.596916Z","shell.execute_reply":"2022-06-29T18:41:21.602595Z"}}
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T18:44:53.049807Z","iopub.execute_input":"2022-06-29T18:44:53.050129Z","iopub.status.idle":"2022-06-29T18:44:53.061527Z","shell.execute_reply.started":"2022-06-29T18:44:53.050082Z","shell.execute_reply":"2022-06-29T18:44:53.059304Z"}}
class FeedBackModel(nn.Module):
    def __init__(self, cfg):
        super(FeedBackModel, self).__init__()
        self.config = torch.load(cfg.config_path)
        self.model = AutoModel.from_config(self.config)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, cfg.target_size)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=False)
        feature = self.pooler(out.last_hidden_state, attention_mask)
        out = self.drop(feature)
        outputs = self.fc(out)
        return outputs, feature

# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T18:43:24.86589Z","iopub.execute_input":"2022-06-29T18:43:24.866217Z","iopub.status.idle":"2022-06-29T18:43:24.875496Z","shell.execute_reply.started":"2022-06-29T18:43:24.866167Z","shell.execute_reply":"2022-06-29T18:43:24.87477Z"}}
@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    PREDS = []
    embeds = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)

        outputs, embeddings = model(ids, mask)
        outputs = F.softmax(outputs, dim=1)
        PREDS.append(outputs.cpu().detach().numpy())
        embeds.append(embeddings.cpu().detach().numpy())
    PREDS = np.concatenate(PREDS)
    embeds = np.concatenate(embeds)
    gc.collect()

    return PREDS, embeds


# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T18:44:29.302341Z","iopub.execute_input":"2022-06-29T18:44:29.302998Z","iopub.status.idle":"2022-06-29T18:44:29.310909Z","shell.execute_reply.started":"2022-06-29T18:44:29.302944Z","shell.execute_reply":"2022-06-29T18:44:29.309698Z"}}
def inference(dataloader, device, model_fold):
    final_preds = []
    embeds = []
    for fold in [model_fold]:
        model = FeedBackModel(CFG)
        state = torch.load(CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                           map_location=device)
        model.load_state_dict(state)
        model.to(device)
        print(f"Getting predictions for model {CFG.model} for fold {fold}")
        preds, embeddings = valid_fn(model, dataloader, device)
        print(preds.shape, embeddings.shape)
        #embeddings = torch.nn.functional.avg_pool1d(embeddings, kernel_size=2)
        final_preds.append(preds)
        embeds.append(embeddings)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    embeds = np.concatenate(embeds)

    return final_preds, embeds



def get_model_preds(CFG, test_df, model_fold):
    test_dataset = FeedBackDataset(test_df, tokenizer, max_length=CFG.max_len)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size,
                             collate_fn=collate_fn,
                             num_workers=2, shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, embeds = inference(test_loader, device, model_fold)
    print(preds.shape, embeds.shape)

    return preds, embeds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG:
    num_workers = 4
    path = "../output/debv3-b-nli-1024/"
    #path = '../output/debv3-l-nli-0630/'
    config_path = path + 'config.pth'
    model = "microsoft/deberta-v3-base"
    #model = "microsoft/deberta-v3-large"
    batch_size = 32
    fc_dropout = 0.2
    target_size = 3
    max_len = 512
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    trn_fold = [0]


CFGS = [CFG] #, CFG2, CFG6, CFG7, CFG11]

def get_nli(discourse_type, discourse_text, topic):
    discourse_text = discourse_text.replace("\n", " ")
    discourse_text = str(discourse_text).lower()
    nli_string = f'Is  this an effective argument for topic: {topic} ? {discourse_type} : {discourse_text} ' # an effective argument?' # for a student essay?'
    #nli_string = f'{discourse_type} : {discourse_text}' # for a student essay?'
    return nli_string

# %% [code] {"execution":{"iopub.status.busy":"2022-06-29T18:41:21.19467Z","iopub.execute_input":"2022-06-29T18:41:21.194982Z","iopub.status.idle":"2022-06-29T18:41:21.211575Z","shell.execute_reply.started":"2022-06-29T18:41:21.194939Z","shell.execute_reply":"2022-06-29T18:41:21.209055Z"}}
class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df = self.df
        discourse_text = df.iloc[index]['discourse_text']
        discourse_text = discourse_text.replace("\n", " ")

        discourse_type = df.iloc[index]['discourse_type']

        # context = df.iloc[index]['context']
        # context = context.replace("\n", " ")

        #essay = df.iloc[index]['essay_text']
        #essay = essay.replace("\n", " ")

        # Map of possible context based on competition rubric
        dt_map = {'Lead': ['Position'],
                  'Position': ['Lead'],
                  'Evidence': ['Claim'],
                  'Claim': ['Position'],
                  'Counterclaim': ['Position'],
                  'Rebuttal': ['Position', 'Counterclaim'],
                  'Concluding Statement': ['Position', 'Claim']
                  }

        text = 'Is ' + discourse_type.lower() + ': ' + discourse_text.rstrip().lower() + ' an effective argument? '
        context = ''
        for dt in dt_map[discourse_type]:
            if dt != discourse_type:
                dt_string = str(df.iloc[index][dt]).rstrip().lower()
                if len(dt_string) > 5:
                    dt_string = dt_string.replace("\n", " ")
                    context += ' ' + self.tokenizer.sep_token + ' ' + dt.lower() + ': ' + self.tokenizer.sep_token + ' ' + dt_string

        inputs = self.tokenizer.encode_plus(
            text,
            context,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            # padding='max_length'
        )

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            # 'target': self.targets[index]
        }


TRAIN_DIR = "../datasets/feedback-prize-effectiveness/train"

def get_essay(essay_id):
    essay_path = os.path.join(TRAIN_DIR, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text



# Bigrams Frequency in String
# Using Counter() + generator expression
def get_bigram_freq(s):
    res = Counter(s[idx: idx + 2] for idx in range(len(s) - 1))
    freq= str(dict(res))
    #print(freq)
    return freq


def get_text_position(s):
    discourse = s['discourse_text'].strip()[0:10]
    essay = s['essay_text']
    pos = essay.find(discourse)
    if pos > 0:
        pos = len(s['essay_text'])/pos
    return pos


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



def get_features(df):
    df['num_discourse_words'] = df['discourse_text'].apply(lambda x: len(x.split()))
    df['num_discourse_chars'] = df['discourse_text'].apply(lambda x: len(x))
    df['text_position'] = df.apply(lambda x: get_text_position(x), axis=1)
    df['num_essay_words'] = df['essay_text'].apply(lambda x: len(x.split()))
    df['num_essay_chars'] = df['essay_text'].apply(lambda x: len(x))
    df['freq_of_essay_id'] = df['essay_id'].map(dict(df['essay_id'].value_counts()))

    df['discourse_Adjectives'] = df['discourse_text'].apply(adjectives)
    df['discourse_Verbs'] = df['discourse_text'].apply(verbs)
    df['discourse_Adverbs'] = df['discourse_text'].apply(adverbs)
    df['discourse_Nouns'] = df['discourse_text'].apply(nouns)

    df['count_next_line_essay'] = df['essay_text'].apply(lambda x: x.count("\n\n"))

    #df['num_paragraphs'] = df['discourse_text'].apply(lambda x: x.count('\n'))

    return df

## Main
debug = args.debug

TEXT_COLUMN = 'text'
if 'csv' in args.train_file:
    df = pd.read_csv(args.train_file)
else:
    df = pd.read_parquet(args.train_file)


oof_dfs = []
oof_preds = []
for i, csv in enumerate(args.csvs):
    oof = pd.read_csv(csv)
    oof = oof.sort_values(by='discourse_id').reset_index(drop=True)
    oof = oof[['discourse_id', 'Ineffective', 'Adequate', 'Effective']]
    oof_dfs.append(oof)


MODEL_WEIGHTS = [
    0.05,  # exp2
    0.20,  # exp7
    0.11,  # exp11
    0.22,  # exp16
    0.21,  # exp203
    0.22,  # exp205
]

print(f"sum of weights {np.sum(MODEL_WEIGHTS)}")

submission_df = pd.DataFrame()

pred_dfs = [
    oof_dfs[0],
    oof_dfs[1],
    oof_dfs[2],
    oof_dfs[3],
    oof_dfs[4],
    oof_dfs[5]
]


submission_df["discourse_id"] = pred_dfs[0]["discourse_id"].values

for model_idx, model_preds in enumerate(pred_dfs):
    if model_idx == 0:
        submission_df["Ineffective"] = MODEL_WEIGHTS[model_idx] * model_preds["Ineffective"]
        submission_df["Adequate"] = MODEL_WEIGHTS[model_idx] * model_preds["Adequate"]
        submission_df["Effective"] = MODEL_WEIGHTS[model_idx] * model_preds["Effective"]
    else:
        submission_df["Ineffective"] += MODEL_WEIGHTS[model_idx] * model_preds["Ineffective"]
        submission_df["Adequate"] += MODEL_WEIGHTS[model_idx] * model_preds["Adequate"]
        submission_df["Effective"] += MODEL_WEIGHTS[model_idx] * model_preds["Effective"]

group_df = submission_df
pred_cols = ["Ineffective", "Adequate", "Effective"]
probs_df = submission_df[pred_cols]
idx = probs_df.idxmax(axis=1)
submission_df['pred'] = idx
group_df['pred'] = idx


for model_idx in range(len(oof_dfs)):
    col_map = dict()
    for col in pred_cols:
        col_map[col] = f"model_{model_idx}_{col}"
    oof_dfs[model_idx] = oof_dfs[model_idx].rename(columns=col_map)


oof_df = oof_dfs[0]
oof_df = oof_df.sort_values(by='discourse_id').reset_index(drop=True)

if len(args.csvs) > 1:
    for csv in oof_dfs[1:]:
        oof_df = oof_df.merge(csv, left_on='discourse_id', right_on='discourse_id', how='left')

print(f'OOF shape: {oof_df.shape}')

train_df = pd.read_csv('../datasets/feedback-prize-effectiveness/train.csv')
df = pd.merge(train_df, df, on="essay_id", how="left")
df = pd.merge(df, oof_df, on="discourse_id", how="left")

if debug:
    df = df.iloc[0:args.debug]  # .reset_index(drop=True)

topics_df = pd.read_csv('../datasets/processed/topics.csv')
df = df.merge(topics_df, on='essay_id', how='left')
df['essay_text'] = df['essay_id'].apply(get_essay)

#topic_err_df = pd.read_csv('./summary_topics_tk.csv')
#df = df.merge(topic_err_df, on='topic', how='left')


submission_df = pd.merge(submission_df, df[['discourse_id', 'essay_id', 'discourse_type']], how='left')
submission_df = submission_df[['essay_id', 'discourse_type', 'pred']]
#print(submission_df.head())
#submission_df.to_csv('./level2/sub_df.csv', index=False)
# add HC pred
group_df = pd.read_csv('./level2/group_df.csv')
#df = pd.merge(df, submission_df[['discourse_id', 'pred']], how='left')
#df = pd.merge(df, group_df, how='left')

if args.debug:
    num_folds = 1
else:
    num_folds = 8



if args.add_fts:
    print(f'Adding features')
    df = get_features(df)
    #print(df.head())
    print('Done..')

if args.add_bert:
    from transformers import DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    for i, CFG in enumerate(CFGS):
        for fold in range(8):
            print(f'Bert embeddings for: {CFG.model}')
            fold_df = df[df['fold'] == fold].reset_index(drop=True)
            preds, embeddings = get_model_preds(CFG, fold_df, fold)
            #drop_cols = ['fold']
            cols_to_add = ['essay_id', 'discourse_id', 'fold', 'discourse_effectiveness']
            if args.add_fts:
                cols_to_add += ft_cols
            fold_df = fold_df[cols_to_add]
            fold_df.loc[:, 'Adequate'] = preds[:, 0]
            fold_df.loc[:, 'Effective'] = preds[:, 1]
            fold_df.loc[:, 'Ineffective'] = preds[:, 2]
            #pred_df = pred_df.drop(drop_cols, axis=1)
            # merge with original df
            fold_df = pd.concat([fold_df, pd.DataFrame(embeddings)], axis=1)
            #print(fold_df.head())
            all_df = pd.concat([all_df, fold_df])
            print(all_df.columns)

from joblib import Parallel, delayed
import time
def get_laser_embeddings(df):
   dt = df['discourse_text']
   embeddings1 = laser.embed_sentences(dt,lang='en')
   essay = df['essay_text']
   embeddings2 = laser.embed_sentences(essay, lang='en')
   print(embeddings1.shape)
   print(embeddings2.shape)
   return embeddings1, embeddings2

if args.add_embeddings:
    print(f'Adding Embeddings')
    fin_df = pd.DataFrame()
    start_time = time.time()
    all_df = pd.DataFrame()
    """
    for fold in range(4):
        train_fold_df = df[df['kfold'] != fold].reset_index(drop=True)
        val_fold_df = df[df['kfold'] == fold].reset_index(drop=True)
        for col in ['discourse_text']:
            train_embeddings, val_embeddings = get_text_embeddings(train_fold_df, val_fold_df, column='discourse_text', max_features=23000)
            #print(train_embeddings.shape, val_embeddings.shape)
            # print(train_fold_df.shape)
            train_fold_df = pd.concat([train_fold_df, pd.DataFrame(train_embeddings)], axis=1)
            val_fold_df = pd.concat([val_fold_df, pd.DataFrame(val_embeddings)], axis=1)
            all_df = pd.concat([train_fold_df, val_fold_df]).reset_index(drop=True)
    df = all_df
    for embedding_type in args.embedding_type:
        print(f'Embedding type: {embedding_type}')
        embeddings1, embeddings2 =  df.progress_apply(get_embeddings,
                                                          embedding_type = embedding_type,
                                                          axis=1,
                                                          result_type="expand"
                                                          )
        #print(len(embeddings1))
        print(embeddings1)
        all_df = pd.concat([all_df, pd.DataFrame(embeddings1)], axis=1)
    """
    laser1, laser2 = get_laser_embeddings(df)
    # print(train_embeddings.shape, val_embeddings.shape)
    # print(train_fold_df.shape)
    df = pd.concat([df, pd.DataFrame(laser1)], axis=1)
    df = pd.concat([df, pd.DataFrame(laser2)], axis=1)
    all_df = df
    #"""
    end_time = time.time()
    print(f'Time: {end_time - start_time}')

topic_map = {
	'seagoing luke animals cowboys': 1,
	'driving phone phones cell' :  2,
	 'phones cell cell phones school': 3,
	 'straights state welfare wa' : 4 ,
	 'summer students project projects': 5,
	 'students online school classes': 6,
	 'car cars usage pollution': 7,
	 'cars driverless car driverless cars': 8,
	 'emotions technology facial computer' : 9,
	 'community service community service help': 10,
	 'sports average school students' : 11,
	 'advice people ask multiple': 12,
	 'extracurricular activities activity students': 13,
	 'electoral college electoral college vote':  14 ,
	 'electoral vote college electoral college' : 14 ,
	 'face mars landform aliens' : 15,
     'venus planet author earth': 16,
}

#vote_df = pd.read_csv('./level2/ensemble_vote.csv')
#df = df.merge(vote_df[['discourse_id', 'preds']], how='left')
df['topic'] = df['topic'].map(topic_map)

cols_to_drop = ['discourse_text', 'essay_text']
df = df.drop(columns=cols_to_drop)
print(f'Final shape: {df.shape}')
print(f'Saved output to: level2/{args.out_file}')
os.makedirs('./level2', exist_ok=True)
df.to_csv('./level2/' + args.out_file, index=False)
