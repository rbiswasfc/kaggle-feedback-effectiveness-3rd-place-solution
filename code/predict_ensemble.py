

# basics
import os
import gc
import sys
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain

# Processing
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# huggingface
from datasets import Dataset
from accelerate import Accelerator
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout

# misc
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ipython
from IPython.display import display
from IPython.core.debugger import set_trace
from tokenizers import AddedToken

# %% [markdown]
# # Enable other models

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:04:20.313026Z","iopub.execute_input":"2022-08-23T12:04:20.314143Z","iopub.status.idle":"2022-08-23T12:04:20.320772Z","shell.execute_reply.started":"2022-08-23T12:04:20.314098Z","shell.execute_reply":"2022-08-23T12:04:20.319256Z"}}
# debv3-l 8 fold
use_exp1 = True

# debv3-l multihead lstm
use_exp3 = False

# debv3-l resolved data
use_exp4 = False

# dexl
use_exp6 = False

# debl kd from dexl
use_exp8 = False

# debv3-l revist
use_exp10 = False

# debv3-l uda
use_exp11 = False

# debv3-l 10 fold
use_exp16 = True

# v3L+b
use_exp102 = False

# debv3-l prompt 8 fold
use_exp205 = False

# debv3-l prompt 10 fold LB 0.565
use_exp209 = True

# longformer
use_exp212 = True

use_exp214 = False

use_full_data_models = True

# enable for running sampled (3k) train vs test
debug = False

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:04:20.322119Z","iopub.execute_input":"2022-08-23T12:04:20.322705Z","iopub.status.idle":"2022-08-23T12:04:21.004264Z","shell.execute_reply.started":"2022-08-23T12:04:20.322671Z","shell.execute_reply":"2022-08-23T12:04:21.003397Z"}}
import pickle
from textblob import TextBlob


# functions for separating the POS Tags
def adjectives(text):
    blob = TextBlob(text)
    return len([word for (word, tag) in blob.tags if tag == 'JJ'])


def verbs(text):
    blob = TextBlob(text)
    return len([word for (word, tag) in blob.tags if tag.startswith('VB')])


def adverbs(text):
    blob = TextBlob(text)
    return len([word for (word, tag) in blob.tags if tag.startswith('RB')])


def nouns(text):
    blob = TextBlob(text)
    return len([word for (word, tag) in blob.tags if tag.startswith('NN')])


# %% [markdown]
# # Load Data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:04:21.007779Z","iopub.execute_input":"2022-08-23T12:04:21.008084Z","iopub.status.idle":"2022-08-23T12:04:23.103267Z","shell.execute_reply.started":"2022-08-23T12:04:21.008051Z","shell.execute_reply":"2022-08-23T12:04:23.099251Z"}}
# Read in test data and assign uid for tracking discourse elements
if debug:
    test_df = pd.read_csv("../datasets/feedback-prize-effectiveness/train.csv")
    test_df = test_df.sample(n=3000).reset_index(drop=True)
else:
    test_df = pd.read_csv("../datasets/feedback-prize-effectiveness/test.csv")

all_ids = test_df["discourse_id"].unique().tolist()
discourse2idx = {discourse: pos for pos, discourse in enumerate(all_ids)}
idx2discourse = {v: k for k, v in discourse2idx.items()}
test_df["uid"] = test_df["discourse_id"].map(discourse2idx)


# Load test essays
def _load_essay(essay_id):
    if debug:
        filename = os.path.join("../datasets/feedback-prize-effectiveness/train", f"{essay_id}.txt")
    else:
        filename = os.path.join("../datasets/feedback-prize-effectiveness/test", f"{essay_id}.txt")
    with open(filename, "r") as f:
        text = f.read()
    return [essay_id, text]


def read_essays(essay_ids, num_jobs=12):
    train_essays = []
    results = Parallel(n_jobs=num_jobs, verbose=1)(delayed(_load_essay)(essay_id) for essay_id in essay_ids)
    for result in results:
        train_essays.append(result)

    result_dict = dict()
    for e in train_essays:
        result_dict[e[0]] = e[1]

    essay_df = pd.Series(result_dict).reset_index()
    essay_df.columns = ["essay_id", "essay_text"]
    return essay_df


essay_ids = test_df["essay_id"].unique().tolist()
essay_df = read_essays(essay_ids)

# Display sample test data
display(test_df.sample())

# %% [markdown]
# # Topics

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:04:23.106655Z","iopub.execute_input":"2022-08-23T12:04:23.107054Z","iopub.status.idle":"2022-08-23T12:05:16.51395Z","shell.execute_reply.started":"2022-08-23T12:04:23.107013Z","shell.execute_reply":"2022-08-23T12:05:16.513042Z"}}
import sys
from bertopic import BERTopic
import glob, pandas as pd, numpy as np, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm

topic_model = BERTopic.load("../models/topic_model/feedback_2021_topic_model")
topic_meta_df = pd.read_csv('../models/topic_model/topic_model_metadata.csv')

# topic_model = BERTopic.load("../models/fdbk-topic-model/feedback_2021_topic_model")
# topic_meta_df = pd.read_csv('../models/fdbk-topic-model/topic_model_metadata.csv')


topic_meta_df = topic_meta_df.rename(columns={'Topic': 'topic', 'Name': 'topic_name'}).drop(columns=['Count'])
topic_meta_df.topic_name = topic_meta_df.topic_name.apply(lambda n: ' '.join(n.split('_')[1:]))

sws = stopwords.words("english") + ["n't", "'s", "'ve"]
fls = glob.glob("../datasets/feedback-prize-effectiveness/test/*.txt")
docs = []
for fl in tqdm(fls):
    with open(fl) as f:
        txt = f.read()
        word_tokens = word_tokenize(txt)
        txt = " ".join([w for w in word_tokens if not w.lower() in sws])
    docs.append(txt)

topics, probs = topic_model.transform(docs)

pred_topics = pd.DataFrame()
dids = list(map(lambda fl: fl.split("/")[-1].split(".")[0], fls))
pred_topics["id"] = dids
pred_topics["topic"] = topics
pred_topics['prob'] = probs
pred_topics = pred_topics.drop(columns={'prob'})
pred_topics = pred_topics.rename(columns={'id': 'essay_id'})

pred_topics = pred_topics.merge(topic_meta_df, left_on='topic', right_on='topic', how='left')
pred_topics.rename(columns={'topic': 'topic_num', 'topic_name': 'topic'}, inplace=True)
pred_topics

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:05:16.515458Z","iopub.execute_input":"2022-08-23T12:05:16.515813Z","iopub.status.idle":"2022-08-23T12:05:16.533773Z","shell.execute_reply.started":"2022-08-23T12:05:16.515779Z","shell.execute_reply":"2022-08-23T12:05:16.532997Z"}}
topic_map = {
    'seagoing luke animals cowboys': 'Should you join the Seagoing Cowboys program?',
    'driving phone phones cell': 'Should drivers be allowed to use cell phones while driving?',
    'phones cell cell phones school': 'Should students be allowed to use cell phones in school?',
    'straights state welfare wa': ' State welfare',
    'summer students project projects': 'Should school summer projects be designed by students or teachers?',
    'students online school classes': 'Is distance learning or online schooling beneficial to students?',
    'car cars usage pollution': 'Should car usage be limited to help reduce pollution?',
    'cars driverless car driverless cars': 'Are driverless cars going to be helpful?',
    'emotions technology facial computer': 'Should computers read the emotional expressions of students in a classroom?',
    'community service community service help': 'Should community service be mandatory for all students?',
    'sports average school students': 'Should students be allowed to participate in sports  unless they have at least a grade B average?',
    'advice people ask multiple': 'Should you ask multiple people for advice?',
    'extracurricular activities activity students': 'Should all students participate in at least one extracurricular activity?',
    'electoral college electoral college vote': 'Should the electoral college be abolished in favor of popular vote?',
    'electoral vote college electoral college': 'Should the electoral college be abolished in favor of popular vote?',
    'face mars landform aliens': 'Is the face on Mars  a natural landform or made by Aliens?',
    'venus planet author earth': 'Is Studying Venus a worthy pursuit?',
}
essay_df = essay_df.merge(pred_topics, on='essay_id', how='left')
essay_df['prompt'] = essay_df['topic'].map(topic_map)

essay_df.head()

# %% [markdown]
# # EXP 19 Model - DEXL

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:16.680906Z","iopub.execute_input":"2022-08-23T12:05:16.681502Z","iopub.status.idle":"2022-08-23T12:05:16.686484Z","shell.execute_reply.started":"2022-08-23T12:05:16.681467Z","shell.execute_reply":"2022-08-23T12:05:16.685673Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/tapt-fpe-dexl",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 5,
    "dropout": 0.1,
    "infer_bs": 8
}
"""
config = json.loads(config)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:16.688501Z","iopub.execute_input":"2022-08-23T12:05:16.688975Z","iopub.status.idle":"2022-08-23T12:05:16.759775Z","shell.execute_reply.started":"2022-08-23T12:05:16.68894Z","shell.execute_reply":"2022-08-23T12:05:16.758674Z"}}
import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION] [COUNTER_CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END] [CLAIM_END]')}")

    print("==" * 40)
    return tokenizer


# --------------- Processing ---------------------------------------------#

TOKEN_MAP = {
    "Lead": ["Lead [LEAD]", "[LEAD_END]"],
    "Position": ["Position [POSITION]", "[POSITION_END]"],
    "Claim": ["Claim [CLAIM]", "[CLAIM_END]"],
    "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM_END]"],
    "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL_END]"],
    "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE_END]"],
    "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT_END]"]
}

DISCOURSE_START_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

DISCOURSE_END_TOKENS = [
    "[LEAD_END]",
    "[POSITION_END]",
    "[CLAIM_END]",
    "[COUNTER_CLAIM_END]",
    "[REBUTTAL_END]",
    "[EVIDENCE_END]",
    "[CONCLUDING_STATEMENT_END]"
]


# NEW_TOKENS = [
#     "[LEAD]",
#     "[POSITION]",
#     "[CLAIM]",
#     "[COUNTER_CLAIM]",
#     "[REBUTTAL]",
#     "[EVIDENCE]",
#     "[CONCLUDING_STATEMENT]",
#     "[LEAD_END]",
#     "[POSITION_END]",
#     "[CLAIM_END]",
#     "[COUNTER_CLAIM_END]",
#     "[REBUTTAL_END]",
#     "[EVIDENCE_END]",
#     "[CONCLUDING_STATEMENT_END]",
#     "[SOE]",
#     "[EOE]",
# ]


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    for cur_discourse in discourse_list:
        if cur_discourse not in to_return:
            to_return[cur_discourse] = []

        matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
        for match in matches:
            span_start, span_end = match.span()
            if span_end <= reading_head:
                continue
            to_return[cur_discourse].append(match.span())
            reading_head = span_end
            break

    # post process
    for cur_discourse in discourse_list:
        if not to_return[cur_discourse]:
            print("resorting to relaxed search...")
            to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    essay_text = "[SOE]" + essay_text + "[EOE]"
    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")
    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text"]].apply(
        lambda x: process_essay(x[0], x[1], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(
            list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


# --------------- Dataset ----------------------------------------------#


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
        }

        self.discourse_type2id = {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)

        # print("adding new tokens...")
        # tokens_to_add = []
        # for this_tok in NEW_TOKENS:
        #     tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
        # self.tokenizer.add_tokens(tokens_to_add)
        print(f"tokenizer len: {len(self.tokenizer)}")

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        self.global_tokens = self.discourse_token_ids.union(self.discourse_end_ids)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset


# --------------- dataset with truncation ---------------------------------------------#


def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

    original_dataset = deepcopy(task_dataset)
    tokenizer = dataset_creator.tokenizer
    START_IDS = dataset_creator.discourse_token_ids
    END_IDS = dataset_creator.discourse_end_ids
    GLOBAL_IDS = dataset_creator.global_tokens

    def tokenize_with_truncation(examples):
        tz = tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=config["max_length"],
            stride=config["stride"],
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )
        return tz

    def process_span(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        buffer = 25  # do not include a head if it is within buffer distance away from last token

        for example_input_ids, example_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            # ------------------- Span Heads -----------------------------------------#
            if len(example_input_ids) < config["max_length"]:  # no truncation
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in START_IDS]
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                        (this_id in START_IDS) & (pos <= config["max_length"] - buffer))]

            n_heads = len(head_candidate)

            # ------------------- Span Tails -----------------------------------------#
            tail_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in END_IDS]

            # ------------------- Edge Cases -----------------------------------------#
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(tail_candidate) > 0) & (len(head_candidate) > 0):
                if tail_candidate[0] < head_candidate[0]:  # truncation effect
                    # print(f"check: heads: {head_candidate}, tails {tail_candidate}")
                    tail_candidate = tail_candidate[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(tail_candidate) < n_heads:
                assert len(tail_candidate) + 1 == n_heads
                assert len(example_input_ids) == config["max_length"]  # should only happen if input text is truncated
                tail_candidate.append(config["max_length"] - 2)  # the token before [SEP] token

            # 3. Additional tails remain in the buffer region
            if len(tail_candidate) > len(head_candidate):
                tail_candidate = tail_candidate[:len(head_candidate)]

            # ------------------- Create the fields ------------------------------------#
            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in head_candidate]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in tail_candidate]

            span_head_idxs.append(head_candidate)
            span_tail_idxs.append(tail_candidate)
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def enforce_alignment(examples):
        uids = []

        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_uids = original_example["uids"]
            char2uid = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_uids)}
            current_example_uids = [char2uid[char_idx] for char_idx in example_span_head_char_start_idxs]
            uids.append(current_example_uids)
        return {"uids": uids}

    def recompute_labels(examples):
        labels = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_labels = original_example["labels"]
            char2label = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_labels)}
            current_example_labels = [char2label[char_idx] for char_idx in example_span_head_char_start_idxs]
            labels.append(current_example_labels)
        return {"labels": labels}

    def recompute_discourse_type_ids(examples):
        discourse_type_ids = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_discourse_type_ids = original_example["discourse_type_ids"]
            char2discourse_id = {k: v for k, v in zip(
                original_example_span_head_char_start_idxs, original_example_discourse_type_ids)}
            current_example_discourse_type_ids = [char2discourse_id[char_idx]
                                                  for char_idx in example_span_head_char_start_idxs]
            discourse_type_ids.append(current_example_discourse_type_ids)
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"head idxs: {head_idxs}, tail idxs {tail_idxs}"

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)
    task_dataset = task_dataset.map(recompute_discourse_type_ids, batched=True)
    task_dataset = task_dataset.map(sanity_check_head_tail, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    if mode != "infer":
        task_dataset = task_dataset.map(recompute_labels, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:16.761343Z","iopub.execute_input":"2022-08-23T12:05:16.761915Z","iopub.status.idle":"2022-08-23T12:05:17.588516Z","shell.execute_reply.started":"2022-08-23T12:05:16.761767Z","shell.execute_reply":"2022-08-23T12:05:17.587706Z"}}
# Reuse for exp7,6,8
os.makedirs(config["model_dir"], exist_ok=True)

print("creating the inference datasets...")
infer_ds_dict = get_fast_dataset(config, test_df, essay_df, mode="infer")
tokenizer = infer_ds_dict["tokenizer"]
infer_dataset = infer_ds_dict["dataset"]
print(infer_dataset)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:17.5899Z","iopub.execute_input":"2022-08-23T12:05:17.590464Z","iopub.status.idle":"2022-08-23T12:05:17.609026Z","shell.execute_reply.started":"2022-08-23T12:05:17.590424Z","shell.execute_reply":"2022-08-23T12:05:17.608062Z"}}
config["len_tokenizer"] = len(tokenizer)

infer_dataset = infer_dataset.sort("input_length")

infer_dataset.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
             'span_tail_idxs', 'discourse_type_ids', 'uids']
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:17.610945Z","iopub.execute_input":"2022-08-23T12:05:17.611325Z","iopub.status.idle":"2022-08-23T12:05:17.629022Z","shell.execute_reply.started":"2022-08-23T12:05:17.611289Z","shell.execute_reply":"2022-08-23T12:05:17.627594Z"}}
from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = 512
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in span_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in
                                       discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        # multitask labels
        def _get_additional_labels(label_id):
            if label_id == 0:
                vec = [0, 0]
            elif label_id == 1:
                vec = [1, 0]
            elif label_id == 2:
                vec = [1, 1]
            elif label_id == -1:
                vec = [-1, -1]
            else:
                raise
            return vec

        if labels is not None:
            additional_labels = []
            for ex_labels in batch["labels"]:
                ex_additional_labels = [_get_additional_labels(el) for el in ex_labels]
                additional_labels.append(ex_additional_labels)
            batch["multitask_labels"] = additional_labels
        # pdb.set_trace()

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:17.630865Z","iopub.execute_input":"2022-08-23T12:05:17.631647Z","iopub.status.idle":"2022-08-23T12:05:17.642881Z","shell.execute_reply.started":"2022-08-23T12:05:17.631605Z","shell.execute_reply":"2022-08-23T12:05:17.641994Z"}}
data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

infer_dl = DataLoader(
    infer_dataset,
    batch_size=config["infer_bs"],
    shuffle=False,
    collate_fn=data_collector
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:17.645996Z","iopub.execute_input":"2022-08-23T12:05:17.646303Z","iopub.status.idle":"2022-08-23T12:05:17.66438Z","shell.execute_reply.started":"2022-08-23T12:05:17.646277Z","shell.execute_reply":"2022-08-23T12:05:17.663539Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": self.base_model.config.num_attention_heads,
                "hidden_size": self.base_model.config.hidden_size,
                "attention_probs_dropout_prob": self.base_model.config.attention_probs_dropout_prob,
                "is_decoder": False,

            }
        )
        self.fpe_span_attention = BertAttention(attention_config, position_embedding_type="relative_key")

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.num_labels = self.config["num_labels"]
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, token_type_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask,
                **kwargs):
        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attention mechanism
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        feature_vector = self.fpe_span_attention(mean_feature_vector, extended_span_attention_mask)[0]

        feature_vector = self.dropout(feature_vector)  # span-atten
        logits = self.classifier(feature_vector)

        ######

        logits = logits[:, :, :3]  # main logits
        return logits


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:17.665867Z","iopub.execute_input":"2022-08-23T12:05:17.666863Z","iopub.status.idle":"2022-08-23T12:05:17.676908Z","shell.execute_reply.started":"2022-08-23T12:05:17.666826Z","shell.execute_reply":"2022-08-23T12:05:17.675922Z"}}
checkpoints = [
    "../models/exp-19-dexl-revisit-part-1/fpe_model_fold_0_best.pth.tar",
     "../models/exp-19-dexl-revisit-part-1/fpe_model_fold_1_best.pth.tar",
    "../models/exp-19-dexl-revisit-part-1/fpe_model_fold_2_best.pth.tar",
    "../models/exp-19-dexl-revisit-part-1/fpe_model_fold_3_best.pth.tar",
    "../models/exp-19-dexl-revisit-part-2/fpe_model_fold_4_best.pth.tar",
    "../models/exp-19-dexl-revisit-part-2/fpe_model_fold_5_best.pth.tar",
    "../models/exp-19-dexl-revisit-part-2/fpe_model_fold_6_best.pth.tar",
    "../models/exp-19-dexl-revisit-part-2/fpe_model_fold_7_best.pth.tar",
]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:05:17.680306Z","iopub.execute_input":"2022-08-23T12:05:17.680573Z","iopub.status.idle":"2022-08-23T12:13:24.288707Z","shell.execute_reply.started":"2022-08-23T12:05:17.680547Z","shell.execute_reply":"2022-08-23T12:13:24.287285Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp19_dexl_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
# del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
gc.collect()
torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:13:24.291718Z","iopub.execute_input":"2022-08-23T12:13:24.292804Z","iopub.status.idle":"2022-08-23T12:13:24.398019Z","shell.execute_reply.started":"2022-08-23T12:13:24.292726Z","shell.execute_reply":"2022-08-23T12:13:24.397166Z"}}
import glob
import pandas as pd

csvs = glob.glob("exp19_dexl_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp19_df = pd.DataFrame()
exp19_df["discourse_id"] = idx
exp19_df["Ineffective"] = preds[:, 0]
exp19_df["Adequate"] = preds[:, 1]
exp19_df["Effective"] = preds[:, 2]

exp19_df = exp19_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:13:24.399235Z","iopub.execute_input":"2022-08-23T12:13:24.39984Z","iopub.status.idle":"2022-08-23T12:13:24.420253Z","shell.execute_reply.started":"2022-08-23T12:13:24.399793Z","shell.execute_reply":"2022-08-23T12:13:24.419038Z"}}
exp19_df.head()

# %% [markdown]
# ### DEXL all data

checkpoints = [
    "../models/exp-19f-dexl-revisit-all-data/fpe_model_all_data_seed_464.pth.tar",
    "../models/exp-19f-dexl-revisit-all-data/fpe_model_all_data_seed_446.pth.tar",
]


def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"19f_dexl_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
gc.collect()
torch.cuda.empty_cache()

import glob
import pandas as pd

csvs = glob.glob("19f_dexl_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp19f_df = pd.DataFrame()
exp19f_df["discourse_id"] = idx
exp19f_df["Ineffective"] = preds[:, 0]
exp19f_df["Adequate"] = preds[:, 1]
exp19f_df["Effective"] = preds[:, 2]

exp19f_df = exp19f_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

exp19f_df.head()

# %% [markdown]
# ### DEL KD All Data

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:15:23.041543Z","iopub.execute_input":"2022-08-23T12:15:23.042086Z","iopub.status.idle":"2022-08-23T12:15:57.745344Z","shell.execute_reply.started":"2022-08-23T12:15:23.042048Z","shell.execute_reply":"2022-08-23T12:15:57.744461Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/tapt-fpe-del-wiki",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 8,

    "use_multitask": true,
    "num_additional_labels": 2
}
"""
config = json.loads(config)
config["len_tokenizer"] = len(tokenizer)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention


# -------- Model ------------------------------------------------------------------#

class FeedbackModel(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": self.base_model.config.num_attention_heads,
                "hidden_size": self.base_model.config.hidden_size,
                "attention_probs_dropout_prob": self.base_model.config.attention_probs_dropout_prob,
                "is_decoder": False,

            }
        )
        self.fpe_span_attention = BertAttention(attention_config, position_embedding_type="relative_key")

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(feature_size, self.num_labels)

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask,
            span_head_idxs,
            span_tail_idxs,
            span_attention_mask,
            labels=None,
            multitask_labels=None,
            **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attention mechanism
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        feature_vector = self.fpe_span_attention(mean_feature_vector, extended_span_attention_mask)[0]

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        logits = logits[:, :, :3]

        return logits


checkpoints = [
    "../models/exp-20-del-kd-all-data-train/fpe_model_kd_seed_1.pth.tar",
    #     "../models/exp-20-del-kd-all-data-train/fpe_model_kd_seed_2.pth.tar",
]


def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp20_del_kd_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
gc.collect()
torch.cuda.empty_cache()

import glob
import pandas as pd

csvs = glob.glob("exp20_del_kd_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp20f_df = pd.DataFrame()
exp20f_df["discourse_id"] = idx
exp20f_df["Ineffective"] = preds[:, 0]
exp20f_df["Adequate"] = preds[:, 1]
exp20f_df["Effective"] = preds[:, 2]

exp20f_df = exp20f_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

exp20f_df.head()

# %% [markdown]
# # EXP212 - Fast Model - Longformer

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:18.907643Z","iopub.execute_input":"2022-08-23T12:16:18.908222Z","iopub.status.idle":"2022-08-23T12:16:18.9139Z","shell.execute_reply.started":"2022-08-23T12:16:18.908157Z","shell.execute_reply":"2022-08-23T12:16:18.912725Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/exp212-longformer-l-prompt-mlm50/mlm_model",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 8
}
"""
config = json.loads(config)

# %% [markdown]
# ## Dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:20.394098Z","iopub.execute_input":"2022-08-23T12:16:20.394545Z","iopub.status.idle":"2022-08-23T12:16:20.521111Z","shell.execute_reply.started":"2022-08-23T12:16:20.394506Z","shell.execute_reply":"2022-08-23T12:16:20.520277Z"}}
import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION] [COUNTER_CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END] [CLAIM_END]')}")

    print("==" * 40)
    return tokenizer


# --------------- Processing ---------------------------------------------#


DISCOURSE_START_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

TOKEN_MAP = {
    "topic": ["Topic [TOPIC]", "[TOPIC END]"],
    "Lead": ["Lead [LEAD]", "[LEAD END]"],
    "Position": ["Position [POSITION]", "[POSITION END]"],
    "Claim": ["Claim [CLAIM]", "[CLAIM END]"],
    "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM END]"],
    "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL END]"],
    "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE END]"],
    "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT END]"]
}

DISCOURSE_END_TOKENS = [
    "[LEAD END]",
    "[POSITION END]",
    "[CLAIM END]",
    "[COUNTER_CLAIM END]",
    "[REBUTTAL END]",
    "[EVIDENCE END]",
    "[CONCLUDING_STATEMENT END]",
]


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    for cur_discourse in discourse_list:
        if cur_discourse not in to_return:
            to_return[cur_discourse] = []

        matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
        for match in matches:
            span_start, span_end = match.span()
            if span_end <= reading_head:
                continue
            to_return[cur_discourse].append(match.span())
            reading_head = span_end
            break

    # post process
    for cur_discourse in discourse_list:
        if not to_return[cur_discourse]:
            print("resorting to relaxed search...")
            to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, prompt, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    essay_text = "[SOE]" + " [TOPIC] " + prompt + " [TOPIC END] " + essay_text + "[EOE]"
    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")
    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text", "prompt"]].apply(
        lambda x: process_essay(x[0], x[1], x[2], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(
            list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


# --------------- Dataset ----------------------------------------------#


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
        }

        self.discourse_type2id = {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)

        # print("adding new tokens...")
        # tokens_to_add = []
        # for this_tok in NEW_TOKENS:
        #     tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
        # self.tokenizer.add_tokens(tokens_to_add)
        print(f"tokenizer len: {len(self.tokenizer)}")

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        self.global_tokens = self.discourse_token_ids.union(self.discourse_end_ids)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset


# --------------- dataset with truncation ---------------------------------------------#


def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

    original_dataset = deepcopy(task_dataset)
    tokenizer = dataset_creator.tokenizer
    START_IDS = dataset_creator.discourse_token_ids
    END_IDS = dataset_creator.discourse_end_ids
    GLOBAL_IDS = dataset_creator.global_tokens

    def tokenize_with_truncation(examples):
        tz = tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=config["max_length"],
            stride=config["stride"],
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )
        return tz

    def process_span(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        buffer = 25  # do not include a head if it is within buffer distance away from last token

        for example_input_ids, example_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            # ------------------- Span Heads -----------------------------------------#
            if len(example_input_ids) < config["max_length"]:  # no truncation
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in START_IDS]
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                        (this_id in START_IDS) & (pos <= config["max_length"] - buffer))]

            n_heads = len(head_candidate)

            # ------------------- Span Tails -----------------------------------------#
            tail_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in END_IDS]

            # ------------------- Edge Cases -----------------------------------------#
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(tail_candidate) > 0) & (len(head_candidate) > 0):
                if tail_candidate[0] < head_candidate[0]:  # truncation effect
                    # print(f"check: heads: {head_candidate}, tails {tail_candidate}")
                    tail_candidate = tail_candidate[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(tail_candidate) < n_heads:
                assert len(tail_candidate) + 1 == n_heads
                assert len(example_input_ids) == config["max_length"]  # should only happen if input text is truncated
                tail_candidate.append(config["max_length"] - 2)  # the token before [SEP] token

            # 3. Additional tails remain in the buffer region
            if len(tail_candidate) > len(head_candidate):
                tail_candidate = tail_candidate[:len(head_candidate)]

            # ------------------- Create the fields ------------------------------------#
            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in head_candidate]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in tail_candidate]

            span_head_idxs.append(head_candidate)
            span_tail_idxs.append(tail_candidate)
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def get_global_attention_mask(examples):
        global_attention_mask = []
        for example_input_ids in examples["input_ids"]:
            global_attention_mask.append([1 if iid in GLOBAL_IDS else 0 for iid in example_input_ids])
        return {"global_attention_mask": global_attention_mask}

    def enforce_alignment(examples):
        uids = []

        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_uids = original_example["uids"]
            char2uid = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_uids)}
            current_example_uids = [char2uid[char_idx] for char_idx in example_span_head_char_start_idxs]
            uids.append(current_example_uids)
        return {"uids": uids}

    def recompute_labels(examples):
        labels = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_labels = original_example["labels"]
            char2label = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_labels)}
            current_example_labels = [char2label[char_idx] for char_idx in example_span_head_char_start_idxs]
            labels.append(current_example_labels)
        return {"labels": labels}

    def recompute_discourse_type_ids(examples):
        discourse_type_ids = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_discourse_type_ids = original_example["discourse_type_ids"]
            char2discourse_id = {k: v for k, v in zip(
                original_example_span_head_char_start_idxs, original_example_discourse_type_ids)}
            current_example_discourse_type_ids = [char2discourse_id[char_idx]
                                                  for char_idx in example_span_head_char_start_idxs]
            discourse_type_ids.append(current_example_discourse_type_ids)
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"head idxs: {head_idxs}, tail idxs {tail_idxs}"

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)
    task_dataset = task_dataset.map(recompute_discourse_type_ids, batched=True)
    task_dataset = task_dataset.map(get_global_attention_mask, batched=True)

    task_dataset = task_dataset.map(sanity_check_head_tail, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    if mode != "infer":
        task_dataset = task_dataset.map(recompute_labels, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:20.526277Z","iopub.execute_input":"2022-08-23T12:16:20.52834Z","iopub.status.idle":"2022-08-23T12:16:21.450122Z","shell.execute_reply.started":"2022-08-23T12:16:20.528302Z","shell.execute_reply":"2022-08-23T12:16:21.449226Z"}}
if use_exp212:
    os.makedirs(config["model_dir"], exist_ok=True)

    print("creating the inference datasets...")
    infer_ds_dict = get_fast_dataset(config, test_df, essay_df, mode="infer")
    tokenizer = infer_ds_dict["tokenizer"]
    infer_dataset = infer_ds_dict["dataset"]
    print(infer_dataset)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:21.452172Z","iopub.execute_input":"2022-08-23T12:16:21.45274Z","iopub.status.idle":"2022-08-23T12:16:21.465788Z","shell.execute_reply.started":"2022-08-23T12:16:21.452701Z","shell.execute_reply":"2022-08-23T12:16:21.464842Z"}}
if use_exp212:
    config["len_tokenizer"] = len(tokenizer)

    infer_dataset = infer_dataset.sort("input_length")

    infer_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs', 'global_attention_mask',
                 'span_tail_idxs', 'discourse_type_ids', 'uids']
    )

# %% [markdown]
# ## Data Loader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:21.828637Z","iopub.execute_input":"2022-08-23T12:16:21.82922Z","iopub.status.idle":"2022-08-23T12:16:21.847728Z","shell.execute_reply.started":"2022-08-23T12:16:21.829158Z","shell.execute_reply":"2022-08-23T12:16:21.846837Z"}}
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = 512
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]
        global_attention_mask = [feature["global_attention_mask"] for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in span_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in
                                       discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        batch["global_attention_mask"] = [
            ex_global_attention_mask + [0] * (max_len - len(ex_global_attention_mask)) for ex_global_attention_mask in
            global_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        # multitask labels
        def _get_additional_labels(label_id):
            if label_id == 0:
                vec = [0, 0]
            elif label_id == 1:
                vec = [1, 0]
            elif label_id == 2:
                vec = [1, 1]
            elif label_id == -1:
                vec = [-1, -1]
            else:
                raise
            return vec

        if labels is not None:
            additional_labels = []
            for ex_labels in batch["labels"]:
                ex_additional_labels = [_get_additional_labels(el) for el in ex_labels]
                additional_labels.append(ex_additional_labels)
            batch["multitask_labels"] = additional_labels
        # pdb.set_trace()

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:22.364376Z","iopub.execute_input":"2022-08-23T12:16:22.365031Z","iopub.status.idle":"2022-08-23T12:16:22.382003Z","shell.execute_reply.started":"2022-08-23T12:16:22.364994Z","shell.execute_reply":"2022-08-23T12:16:22.380831Z"}}
if use_exp212:
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    infer_dl = DataLoader(
        infer_dataset,
        batch_size=config["infer_bs"],
        shuffle=False,
        collate_fn=data_collector
    )

# %% [markdown]
# ## Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:23.112886Z","iopub.execute_input":"2022-08-23T12:16:23.11343Z","iopub.status.idle":"2022-08-23T12:16:23.129999Z","shell.execute_reply.started":"2022-08-23T12:16:23.113394Z","shell.execute_reply":"2022-08-23T12:16:23.128951Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention


class FeedbackModel(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])

        self.num_labels = self.config["num_labels"]

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": self.base_model.config.num_attention_heads,
                "hidden_size": self.base_model.config.hidden_size,
                "attention_probs_dropout_prob": self.base_model.config.attention_probs_dropout_prob,
                "is_decoder": False,

            }
        )
        self.fpe_span_attention = BertAttention(attention_config, position_embedding_type="relative_key")

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(feature_size, self.num_labels)

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask,
            span_head_idxs,
            span_tail_idxs,
            span_attention_mask,
            global_attention_mask,
            labels=None,
            multitask_labels=None,
            **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            global_attention_mask=global_attention_mask,
        )
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attention mechanism
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        feature_vector = self.fpe_span_attention(mean_feature_vector, extended_span_attention_mask)[0]

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits


# %% [markdown]
# ## Inference

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:23.975449Z","iopub.execute_input":"2022-08-23T12:16:23.976042Z","iopub.status.idle":"2022-08-23T12:16:23.98077Z","shell.execute_reply.started":"2022-08-23T12:16:23.976005Z","shell.execute_reply":"2022-08-23T12:16:23.979886Z"}}
checkpoints = [
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_0_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_1_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_2_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_3_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_4_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_5_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_6_best.pth.tar",
    "../models/exp212-longformer-l-prompt-mlm50/fpe_model_fold_7_best.pth.tar",
]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:16:24.396573Z","iopub.execute_input":"2022-08-23T12:16:24.397686Z","iopub.status.idle":"2022-08-23T12:21:58.55966Z","shell.execute_reply.started":"2022-08-23T12:16:24.397639Z","shell.execute_reply":"2022-08-23T12:21:58.558647Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp212_longformer_model_preds_{model_id}.csv", index=False)


if use_exp212:
    for model_id, checkpoint in enumerate(checkpoints):
        print(f"infering from {checkpoint}")
        model = FeedbackModel(config)
        ckpt = torch.load(checkpoint)
        print(f"model performance on validation set = {ckpt['loss']}")
        model.load_state_dict(ckpt['state_dict'])
        inference_fn(model, infer_dl, model_id)

    del model
    del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
    gc.collect()
    torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:21:58.565079Z","iopub.execute_input":"2022-08-23T12:21:58.567714Z","iopub.status.idle":"2022-08-23T12:21:58.649059Z","shell.execute_reply.started":"2022-08-23T12:21:58.567672Z","shell.execute_reply":"2022-08-23T12:21:58.648144Z"}}
if use_exp212:
    import glob
    import pandas as pd

    csvs = glob.glob("exp212_longformer_model_preds_*.csv")

    idx = []
    preds = []

    for csv_idx, csv in enumerate(csvs):

        print("==" * 40)
        print(f"preds in {csv}")
        df = pd.read_csv(csv)
        df = df.sort_values(by=["discourse_id"])
        print(df.head(10))
        print("==" * 40)

        temp_preds = df.drop(["discourse_id"], axis=1).values
        if csv_idx == 0:
            idx = list(df["discourse_id"])
            preds = temp_preds
        else:
            preds += temp_preds

    preds = preds / len(csvs)

    exp212_df = pd.DataFrame()
    exp212_df["discourse_id"] = idx
    exp212_df["Ineffective"] = preds[:, 0]
    exp212_df["Adequate"] = preds[:, 1]
    exp212_df["Effective"] = preds[:, 2]

    exp212_df = exp212_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:21:58.653313Z","iopub.execute_input":"2022-08-23T12:21:58.654009Z","iopub.status.idle":"2022-08-23T12:21:58.672863Z","shell.execute_reply.started":"2022-08-23T12:21:58.653964Z","shell.execute_reply":"2022-08-23T12:21:58.671856Z"}}
if use_exp212:
    display(exp212_df.head())

# %% [markdown] {"execution":{"iopub.status.busy":"2022-07-17T18:52:10.311607Z","iopub.execute_input":"2022-07-17T18:52:10.311971Z","iopub.status.idle":"2022-07-17T18:52:10.323544Z","shell.execute_reply.started":"2022-07-17T18:52:10.311935Z","shell.execute_reply":"2022-07-17T18:52:10.32276Z"}}
# # EXP1 - delv3  8 fold Fast Model - SPAN MLM 40%

# %% [markdown]
# ## Config

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:21:59.058623Z","iopub.execute_input":"2022-08-23T12:21:59.059007Z","iopub.status.idle":"2022-08-23T12:21:59.066651Z","shell.execute_reply.started":"2022-08-23T12:21:59.058972Z","shell.execute_reply":"2022-08-23T12:21:59.065845Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/tapt-fpe-delv3-span-mlm-04",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 5,
    "dropout": 0.1,
    "infer_bs": 8
}
"""
config = json.loads(config)

# %% [markdown]
# ## Dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:21:59.068949Z","iopub.execute_input":"2022-08-23T12:21:59.069409Z","iopub.status.idle":"2022-08-23T12:21:59.143427Z","shell.execute_reply.started":"2022-08-23T12:21:59.069372Z","shell.execute_reply":"2022-08-23T12:21:59.142358Z"}}
import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END]')}")

    print("==" * 40)
    return tokenizer


# --------------- Processing ---------------------------------------------#
TOKEN_MAP = {
    "Lead": ["Lead [LEAD]", "[LEAD_END]"],
    "Position": ["Position [POSITION]", "[POSITION_END]"],
    "Claim": ["Claim [CLAIM]", "[CLAIM_END]"],
    "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM_END]"],
    "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL_END]"],
    "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE_END]"],
    "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT_END]"]
}

DISCOURSE_START_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

DISCOURSE_END_TOKENS = [
    "[LEAD_END]",
    "[POSITION_END]",
    "[CLAIM_END]",
    "[COUNTER_CLAIM_END]",
    "[REBUTTAL_END]",
    "[EVIDENCE_END]",
    "[CONCLUDING_STATEMENT_END]"
]


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    for cur_discourse in discourse_list:
        if cur_discourse not in to_return:
            to_return[cur_discourse] = []

        matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
        for match in matches:
            span_start, span_end = match.span()
            if span_end <= reading_head:
                continue
            to_return[cur_discourse].append(match.span())
            reading_head = span_end
            break

    # post process
    for cur_discourse in discourse_list:
        if not to_return[cur_discourse]:
            print("resorting to relaxed search...")
            to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    essay_text = "[SOE]" + essay_text + "[EOE]"
    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    # anno_df["discourse_span"] = anno_df[["essay_id", "discourse_text"]].apply(
    #     lambda x: get_substring_span(
    #         notes_df[notes_df["essay_id"] == x[0]].iloc[0].essay_text,
    #         x[1]
    #     ), axis=1
    # )

    # anno_df["discourse_start"] = anno_df["discourse_span"].apply(lambda x: x[0])
    # anno_df["discourse_end"] = anno_df["discourse_span"].apply(lambda x: x[1])

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")
    #     set_trace()
    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text"]].apply(
        lambda x: process_essay(x[0], x[1], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(
            list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


# --------------- Dataset ----------------------------------------------#


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
        }

        self.discourse_type2id = {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset


# --------------- dataset with truncation ---------------------------------------------#


def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

    original_dataset = deepcopy(task_dataset)
    tokenizer = dataset_creator.tokenizer
    START_IDS = dataset_creator.discourse_token_ids
    END_IDS = dataset_creator.discourse_end_ids

    def tokenize_with_truncation(examples):
        tz = tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=config["max_length"],
            stride=config["stride"],
            return_overflowing_tokens=True,
        )
        return tz

    def process_span(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        buffer = 25  # do not include a head if it is within buffer distance away from last token

        for example_input_ids, example_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            # ------------------- Span Heads -----------------------------------------#
            if len(example_input_ids) < config["max_length"]:  # no truncation
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in START_IDS]
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                        (this_id in START_IDS) & (pos <= config["max_length"] - buffer))]

            n_heads = len(head_candidate)

            # ------------------- Span Tails -----------------------------------------#
            tail_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in END_IDS]

            # ------------------- Edge Cases -----------------------------------------#
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(tail_candidate) > 0) & (len(head_candidate) > 0):
                if tail_candidate[0] < head_candidate[0]:  # truncation effect
                    # print(f"check: heads: {head_candidate}, tails {tail_candidate}")
                    tail_candidate = tail_candidate[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(tail_candidate) < n_heads:
                assert len(tail_candidate) + 1 == n_heads
                assert len(example_input_ids) == config["max_length"]  # should only happen if input text is truncated
                tail_candidate.append(config["max_length"] - 2)  # the token before [SEP] token

            # 3. Additional tails remain in the buffer region
            if len(tail_candidate) > len(head_candidate):
                tail_candidate = tail_candidate[:len(head_candidate)]

            # ------------------- Create the fields ------------------------------------#
            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in head_candidate]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in tail_candidate]

            span_head_idxs.append(head_candidate)
            span_tail_idxs.append(tail_candidate)
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def enforce_alignment(examples):
        uids = []

        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_uids = original_example["uids"]
            char2uid = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_uids)}
            current_example_uids = [char2uid[char_idx] for char_idx in example_span_head_char_start_idxs]
            uids.append(current_example_uids)
        return {"uids": uids}

    def recompute_labels(examples):
        labels = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_labels = original_example["labels"]
            char2label = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_labels)}
            current_example_labels = [char2label[char_idx] for char_idx in example_span_head_char_start_idxs]
            labels.append(current_example_labels)
        return {"labels": labels}

    def recompute_discourse_type_ids(examples):
        discourse_type_ids = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_discourse_type_ids = original_example["discourse_type_ids"]
            char2discourse_id = {k: v for k, v in zip(
                original_example_span_head_char_start_idxs, original_example_discourse_type_ids)}
            current_example_discourse_type_ids = [char2discourse_id[char_idx]
                                                  for char_idx in example_span_head_char_start_idxs]
            discourse_type_ids.append(current_example_discourse_type_ids)
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"head idxs: {head_idxs}, tail idxs {tail_idxs}"

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)
    task_dataset = task_dataset.map(recompute_discourse_type_ids, batched=True)
    task_dataset = task_dataset.map(sanity_check_head_tail, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    if mode != "infer":
        task_dataset = task_dataset.map(recompute_labels, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:21:59.145312Z","iopub.execute_input":"2022-08-23T12:21:59.145717Z","iopub.status.idle":"2022-08-23T12:22:00.255959Z","shell.execute_reply.started":"2022-08-23T12:21:59.145681Z","shell.execute_reply":"2022-08-23T12:22:00.254975Z"}}
if use_exp1 or use_exp10 or use_exp11 or use_exp16:
    os.makedirs(config["model_dir"], exist_ok=True)

    print("creating the inference datasets...")
    infer_ds_dict = get_fast_dataset(config, test_df, essay_df, mode="infer")
    tokenizer = infer_ds_dict["tokenizer"]
    infer_dataset = infer_ds_dict["dataset"]
    print(infer_dataset)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.257606Z","iopub.execute_input":"2022-08-23T12:22:00.258155Z","iopub.status.idle":"2022-08-23T12:22:00.272555Z","shell.execute_reply.started":"2022-08-23T12:22:00.258113Z","shell.execute_reply":"2022-08-23T12:22:00.271466Z"}}

if use_exp1 or use_exp10 or use_exp11:
    config["len_tokenizer"] = len(tokenizer)
    infer_dataset = infer_dataset.sort("input_length")
    infer_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
                 'span_tail_idxs', 'discourse_type_ids', 'uids']
    )

# %% [markdown]
# ## Data Loader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.274178Z","iopub.execute_input":"2022-08-23T12:22:00.274637Z","iopub.status.idle":"2022-08-23T12:22:00.292981Z","shell.execute_reply.started":"2022-08-23T12:22:00.274573Z","shell.execute_reply":"2022-08-23T12:22:00.291822Z"}}
from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in span_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in
                                       discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        def _get_additional_labels(label_id):
            if label_id == 0:
                vec = [0, 0]
            elif label_id == 1:
                vec = [1, 0]
            elif label_id == 2:
                vec = [1, 1]
            elif label_id == -1:
                vec = [-1, -1]
            else:
                raise
            return vec

        if labels is not None:
            additional_labels = []
            for ex_labels in batch["labels"]:
                ex_additional_labels = [_get_additional_labels(el) for el in ex_labels]
                additional_labels.append(ex_additional_labels)
            batch["multitask_labels"] = additional_labels

        # batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.294787Z","iopub.execute_input":"2022-08-23T12:22:00.295953Z","iopub.status.idle":"2022-08-23T12:22:00.305319Z","shell.execute_reply.started":"2022-08-23T12:22:00.295913Z","shell.execute_reply":"2022-08-23T12:22:00.304091Z"}}
if use_exp1 or use_exp10 or use_exp11:
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    infer_dl = DataLoader(
        infer_dataset,
        batch_size=config["infer_bs"],
        shuffle=False,
        collate_fn=data_collector
    )

# %% [markdown]
# ## Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.306948Z","iopub.execute_input":"2022-08-23T12:22:00.307446Z","iopub.status.idle":"2022-08-23T12:22:00.324359Z","shell.execute_reply.started":"2022-08-23T12:22:00.307411Z","shell.execute_reply":"2022-08-23T12:22:00.32324Z"}}
import gc
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, DebertaV2Attention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.num_labels = self.config["num_labels"]  # 5
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask, **kwargs):

        bs = input_ids.shape[0]  # batch size
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        mean_feature_vector = []
        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        logits = logits[:, :, :3]  # main logits
        return logits


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.325636Z","iopub.execute_input":"2022-08-23T12:22:00.326406Z","iopub.status.idle":"2022-08-23T12:22:00.337283Z","shell.execute_reply.started":"2022-08-23T12:22:00.326372Z","shell.execute_reply":"2022-08-23T12:22:00.336272Z"}}
def process_swa_checkpoint(checkpoint_path):
    """
    helper function to process swa checkpoints
    """
    ckpt = torch.load(checkpoint_path)

    print("processing ckpt...")
    print("removing module from keys...")
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    processed_state = {"state_dict": new_state_dict}

    # delete old state
    del state_dict
    gc.collect()

    return processed_state


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.340684Z","iopub.execute_input":"2022-08-23T12:22:00.341175Z","iopub.status.idle":"2022-08-23T12:22:00.347035Z","shell.execute_reply.started":"2022-08-23T12:22:00.341099Z","shell.execute_reply":"2022-08-23T12:22:00.34579Z"}}
checkpoints = [
    "../models/a-delv3-prod-8-folds/fpe_model_fold_0_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_1_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_2_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_3_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_4_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_5_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_6_best.pth.tar",
    "../models/a-delv3-prod-8-folds/fpe_model_fold_7_best.pth.tar",
]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:22:00.353776Z","iopub.execute_input":"2022-08-23T12:22:00.35449Z","iopub.status.idle":"2022-08-23T12:27:28.357969Z","shell.execute_reply.started":"2022-08-23T12:22:00.354464Z","shell.execute_reply":"2022-08-23T12:27:28.357141Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp01_delv3_8folds_model_preds_{model_id}.csv", index=False)


if use_exp1:

    for model_id, checkpoint in enumerate(checkpoints):
        print(f"infering from {checkpoint}")
        model = FeedbackModel(config)
        if "swa" in checkpoint:
            ckpt = process_swa_checkpoint(checkpoint)
        else:
            ckpt = torch.load(checkpoint)
            print(f"validation score for fold {model_id} = {ckpt['loss']}")
        model.load_state_dict(ckpt['state_dict'])
        inference_fn(model, infer_dl, model_id)

    del model
    # del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
    gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:27:28.360396Z","iopub.execute_input":"2022-08-23T12:27:28.360681Z","iopub.status.idle":"2022-08-23T12:27:28.415243Z","shell.execute_reply.started":"2022-08-23T12:27:28.360654Z","shell.execute_reply":"2022-08-23T12:27:28.414422Z"}}
if use_exp1:
    import glob
    import pandas as pd

    csvs = glob.glob("exp01_delv3_8folds_model_preds_*.csv")

    idx = []
    preds = []

    for csv_idx, csv in enumerate(csvs):

        print("==" * 40)
        print(f"preds in {csv}")
        df = pd.read_csv(csv)
        df = df.sort_values(by=["discourse_id"])
        print(df.head(10))
        print("==" * 40)

        temp_preds = df.drop(["discourse_id"], axis=1).values
        if csv_idx == 0:
            idx = list(df["discourse_id"])
            preds = temp_preds
        else:
            preds += temp_preds

    preds = preds / len(csvs)

    exp01_df = pd.DataFrame()
    exp01_df["discourse_id"] = idx
    exp01_df["Ineffective"] = preds[:, 0]
    exp01_df["Adequate"] = preds[:, 1]
    exp01_df["Effective"] = preds[:, 2]

    exp01_df = exp01_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:27:28.416837Z","iopub.execute_input":"2022-08-23T12:27:28.417547Z","iopub.status.idle":"2022-08-23T12:27:28.42239Z","shell.execute_reply.started":"2022-08-23T12:27:28.417505Z","shell.execute_reply":"2022-08-23T12:27:28.421472Z"}}
if use_exp1:
    exp01_df.head()


#EXP16
config = """{
    "debug": false,

    "base_model_path": "../models/tapt-fpe-delv3-span-mlm-04",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 8
}
"""
config = json.loads(config)

import gc
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, DebertaV2Attention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.num_labels = self.config["num_labels"]
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask, **kwargs):

        bs = input_ids.shape[0]  # batch size
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        mean_feature_vector = []
        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        logits = logits[:, :, :3]  # main logits
        return logits


checkpoints = [
    "../models/exp-16-part-1/fpe_model_fold_0_best.pth.tar",
    "../models/exp-16-part-1/fpe_model_fold_1_best.pth.tar",
    "../models/exp-16-part-1/fpe_model_fold_2_best.pth.tar",
    "../models/exp-16-part-1/fpe_model_fold_3_best.pth.tar",
    "../models/exp-16-part-1/fpe_model_fold_4_best.pth.tar",
    "../models/exp-16-part-2/fpe_model_fold_5_best.pth.tar",
    "../models/exp-16-part-2/fpe_model_fold_6_best.pth.tar",
    "../models/exp-16-part-2/fpe_model_fold_7_best.pth.tar",
    "../models/exp-16-part-2/fpe_model_fold_8_best.pth.tar",
    "../models/exp-16-part-2/fpe_model_fold_9_best.pth.tar",
]


def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp16_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    if "swa" in checkpoint:
        ckpt = process_swa_checkpoint(checkpoint)
    else:
        ckpt = torch.load(checkpoint)
        print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
# del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
torch.cuda.empty_cache()
gc.collect()

import glob
import pandas as pd

csvs = glob.glob("exp16_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp16_df = pd.DataFrame()
exp16_df["discourse_id"] = idx
exp16_df["Ineffective"] = preds[:, 0]
exp16_df["Adequate"] = preds[:, 1]
exp16_df["Effective"] = preds[:, 2]

exp16_df = exp16_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# All data trained models
import gc
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, DebertaV2Attention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.num_labels = self.config["num_labels"]
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask, **kwargs):

        bs = input_ids.shape[0]  # batch size
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        mean_feature_vector = []
        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        logits = logits[:, :, :3]  # main logits
        return logits


#########################
from copy import deepcopy

checkpoints = [
    "../models/delv3-all-folds/swa_fpe_all_exp_10a_mask_aug.pth.tar",
    "../models/delv3-all-folds/fpe_all_exp_16a_mixout_high_gamma_high_mask_aug.pth.tar",
]


def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp_rb_full_data_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    new_config = deepcopy(config)
    new_config["num_labels"] = 5

    model = FeedbackModel(new_config)
    if "swa" in checkpoint:
        ckpt = process_swa_checkpoint(checkpoint)
    else:
        ckpt = torch.load(checkpoint)
        print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
# del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
torch.cuda.empty_cache()
gc.collect()

import glob
import pandas as pd

csvs = glob.glob("exp_rb_full_data_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp99_rb_all_df = pd.DataFrame()
exp99_rb_all_df["discourse_id"] = idx
exp99_rb_all_df["Ineffective"] = preds[:, 0]
exp99_rb_all_df["Adequate"] = preds[:, 1]
exp99_rb_all_df["Effective"] = preds[:, 2]

exp99_rb_all_df = exp99_rb_all_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(
    np.mean).reset_index()

print(exp99_rb_all_df.head())

# %% [markdown]
# # EXP 213 - Deberta-large 10 fold LB 0.565

# %% [markdown]
# ## Config

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:25.924286Z","iopub.execute_input":"2022-08-23T12:35:25.924567Z","iopub.status.idle":"2022-08-23T12:35:25.93703Z","shell.execute_reply.started":"2022-08-23T12:35:25.924544Z","shell.execute_reply":"2022-08-23T12:35:25.936277Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/deberta-large-prompt-mlm40/",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 8
}
"""
config = json.loads(config)

# %% [markdown]
# ## Dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:25.938884Z","iopub.execute_input":"2022-08-23T12:35:25.939257Z","iopub.status.idle":"2022-08-23T12:35:26.011058Z","shell.execute_reply.started":"2022-08-23T12:35:25.939222Z","shell.execute_reply":"2022-08-23T12:35:26.010047Z"}}
import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION] [COUNTER_CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END] [CLAIM_END]')}")

    print("==" * 40)
    return tokenizer


# --------------- Processing ---------------------------------------------#


DISCOURSE_START_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

TOKEN_MAP = {
    "topic": ["Topic [TOPIC]", "[TOPIC END]"],
    "Lead": ["Lead [LEAD]", "[LEAD END]"],
    "Position": ["Position [POSITION]", "[POSITION END]"],
    "Claim": ["Claim [CLAIM]", "[CLAIM END]"],
    "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM END]"],
    "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL END]"],
    "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE END]"],
    "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT END]"]
}

DISCOURSE_END_TOKENS = [
    "[LEAD END]",
    "[POSITION END]",
    "[CLAIM END]",
    "[COUNTER_CLAIM END]",
    "[REBUTTAL END]",
    "[EVIDENCE END]",
    "[CONCLUDING_STATEMENT END]",
]


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    for cur_discourse in discourse_list:
        if cur_discourse not in to_return:
            to_return[cur_discourse] = []

        matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
        for match in matches:
            span_start, span_end = match.span()
            if span_end <= reading_head:
                continue
            to_return[cur_discourse].append(match.span())
            reading_head = span_end
            break

    # post process
    for cur_discourse in discourse_list:
        if not to_return[cur_discourse]:
            print("resorting to relaxed search...")
            to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, topic, prompt, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    # essay_text = "[SOE]" + " [TOPIC] " + topic + " [TOPIC END] " +  "[PROMPT] " + prompt + " [PROMPT END] " + essay_text + "[EOE]"
    essay_text = "[SOE]" + " [TOPIC] " + prompt + " [TOPIC END] " + essay_text + "[EOE]"

    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")
    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text", "topic", "prompt"]].apply(
        lambda x: process_essay(x[0], x[1], x[2], x[3], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(
            list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


# --------------- Dataset ----------------------------------------------#


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
        }

        self.discourse_type2id = {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)

        # print("adding new tokens...")
        # tokens_to_add = []
        # for this_tok in NEW_TOKENS:
        #     tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
        # self.tokenizer.add_tokens(tokens_to_add)
        print(f"tokenizer len: {len(self.tokenizer)}")

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        self.global_tokens = self.discourse_token_ids.union(self.discourse_end_ids)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset


# --------------- dataset with truncation ---------------------------------------------#


def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

    original_dataset = deepcopy(task_dataset)
    tokenizer = dataset_creator.tokenizer
    START_IDS = dataset_creator.discourse_token_ids
    END_IDS = dataset_creator.discourse_end_ids
    GLOBAL_IDS = dataset_creator.global_tokens

    def tokenize_with_truncation(examples):
        tz = tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=config["max_length"],
            stride=config["stride"],
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )
        return tz

    def process_span(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        buffer = 25  # do not include a head if it is within buffer distance away from last token

        for example_input_ids, example_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            # ------------------- Span Heads -----------------------------------------#
            if len(example_input_ids) < config["max_length"]:  # no truncation
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in START_IDS]
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                        (this_id in START_IDS) & (pos <= config["max_length"] - buffer))]

            n_heads = len(head_candidate)

            # ------------------- Span Tails -----------------------------------------#
            tail_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in END_IDS]

            # ------------------- Edge Cases -----------------------------------------#
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(tail_candidate) > 0) & (len(head_candidate) > 0):
                if tail_candidate[0] < head_candidate[0]:  # truncation effect
                    # print(f"check: heads: {head_candidate}, tails {tail_candidate}")
                    tail_candidate = tail_candidate[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(tail_candidate) < n_heads:
                assert len(tail_candidate) + 1 == n_heads
                assert len(example_input_ids) == config["max_length"]  # should only happen if input text is truncated
                tail_candidate.append(config["max_length"] - 2)  # the token before [SEP] token

            # 3. Additional tails remain in the buffer region
            if len(tail_candidate) > len(head_candidate):
                tail_candidate = tail_candidate[:len(head_candidate)]

            # ------------------- Create the fields ------------------------------------#
            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in head_candidate]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in tail_candidate]

            span_head_idxs.append(head_candidate)
            span_tail_idxs.append(tail_candidate)
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def enforce_alignment(examples):
        uids = []

        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_uids = original_example["uids"]
            char2uid = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_uids)}
            current_example_uids = [char2uid[char_idx] for char_idx in example_span_head_char_start_idxs]
            uids.append(current_example_uids)
        return {"uids": uids}

    def recompute_labels(examples):
        labels = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_labels = original_example["labels"]
            char2label = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_labels)}
            current_example_labels = [char2label[char_idx] for char_idx in example_span_head_char_start_idxs]
            labels.append(current_example_labels)
        return {"labels": labels}

    def recompute_discourse_type_ids(examples):
        discourse_type_ids = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_discourse_type_ids = original_example["discourse_type_ids"]
            char2discourse_id = {k: v for k, v in zip(
                original_example_span_head_char_start_idxs, original_example_discourse_type_ids)}
            current_example_discourse_type_ids = [char2discourse_id[char_idx]
                                                  for char_idx in example_span_head_char_start_idxs]
            discourse_type_ids.append(current_example_discourse_type_ids)
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"head idxs: {head_idxs}, tail idxs {tail_idxs}"

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)
    task_dataset = task_dataset.map(recompute_discourse_type_ids, batched=True)
    task_dataset = task_dataset.map(sanity_check_head_tail, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    if mode != "infer":
        task_dataset = task_dataset.map(recompute_labels, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.01323Z","iopub.execute_input":"2022-08-23T12:35:26.014294Z","iopub.status.idle":"2022-08-23T12:35:26.7974Z","shell.execute_reply.started":"2022-08-23T12:35:26.014247Z","shell.execute_reply":"2022-08-23T12:35:26.796484Z"}}
os.makedirs(config["model_dir"], exist_ok=True)

print("creating the inference datasets...")
infer_ds_dict = get_fast_dataset(config, test_df, essay_df, mode="infer")
tokenizer = infer_ds_dict["tokenizer"]
infer_dataset = infer_ds_dict["dataset"]
print(infer_dataset)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.798698Z","iopub.execute_input":"2022-08-23T12:35:26.79914Z","iopub.status.idle":"2022-08-23T12:35:26.813261Z","shell.execute_reply.started":"2022-08-23T12:35:26.799104Z","shell.execute_reply":"2022-08-23T12:35:26.812424Z"}}
config["len_tokenizer"] = len(tokenizer)

infer_dataset = infer_dataset.sort("input_length")

infer_dataset.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
             'span_tail_idxs', 'discourse_type_ids', 'uids']
)

# %% [markdown]
# ## Data Loader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.814955Z","iopub.execute_input":"2022-08-23T12:35:26.815451Z","iopub.status.idle":"2022-08-23T12:35:26.832391Z","shell.execute_reply.started":"2022-08-23T12:35:26.815416Z","shell.execute_reply":"2022-08-23T12:35:26.83146Z"}}
from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = 512
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in span_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in
                                       discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        # multitask labels
        def _get_additional_labels(label_id):
            if label_id == 0:
                vec = [0, 0]
            elif label_id == 1:
                vec = [1, 0]
            elif label_id == 2:
                vec = [1, 1]
            elif label_id == -1:
                vec = [-1, -1]
            else:
                raise
            return vec

        if labels is not None:
            additional_labels = []
            for ex_labels in batch["labels"]:
                ex_additional_labels = [_get_additional_labels(el) for el in ex_labels]
                additional_labels.append(ex_additional_labels)
            batch["multitask_labels"] = additional_labels
        # pdb.set_trace()

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.835477Z","iopub.execute_input":"2022-08-23T12:35:26.835769Z","iopub.status.idle":"2022-08-23T12:35:26.845846Z","shell.execute_reply.started":"2022-08-23T12:35:26.835744Z","shell.execute_reply":"2022-08-23T12:35:26.845Z"}}
data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

infer_dl = DataLoader(
    infer_dataset,
    batch_size=config["infer_bs"],
    shuffle=False,
    collate_fn=data_collector
)

# %% [markdown]
# ## Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.847257Z","iopub.execute_input":"2022-08-23T12:35:26.847841Z","iopub.status.idle":"2022-08-23T12:35:26.86428Z","shell.execute_reply.started":"2022-08-23T12:35:26.847806Z","shell.execute_reply":"2022-08-23T12:35:26.863429Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": self.base_model.config.num_attention_heads,
                "hidden_size": self.base_model.config.hidden_size,
                "attention_probs_dropout_prob": self.base_model.config.attention_probs_dropout_prob,
                "is_decoder": False,

            }
        )
        self.fpe_span_attention = BertAttention(attention_config, position_embedding_type="relative_key")

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.num_labels = self.config["num_labels"]
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, token_type_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask,
                **kwargs):
        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attention mechanism
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        feature_vector = self.fpe_span_attention(mean_feature_vector, extended_span_attention_mask)[0]

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        ######

        # logits = logits[:,:, :3] # main logits
        return logits


# %% [markdown]
# ## Inference

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.865504Z","iopub.execute_input":"2022-08-23T12:35:26.866031Z","iopub.status.idle":"2022-08-23T12:35:26.877145Z","shell.execute_reply.started":"2022-08-23T12:35:26.865987Z","shell.execute_reply":"2022-08-23T12:35:26.876316Z"}}
checkpoints = [
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_0_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_1_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_2_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_3_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_4_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_5_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_6_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_7_best.pth.tar",
     "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_8_best.pth.tar",
    "../models/exp213-deb-l-prompt-mlm50/fpe_model_fold_9_best.pth.tar",

]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:35:26.878677Z","iopub.execute_input":"2022-08-23T12:35:26.879543Z","iopub.status.idle":"2022-08-23T12:41:09.476548Z","shell.execute_reply.started":"2022-08-23T12:35:26.879508Z","shell.execute_reply":"2022-08-23T12:41:09.475498Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp213_dl_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
# del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# # Exp213a

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:41:09.478836Z","iopub.execute_input":"2022-08-23T12:41:09.48046Z","iopub.status.idle":"2022-08-23T12:41:09.485216Z","shell.execute_reply.started":"2022-08-23T12:41:09.48042Z","shell.execute_reply":"2022-08-23T12:41:09.484383Z"}}
checkpoints = [
    "../models/exp213a-deb-l-prompt/fpe_model_fold_0_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_1_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_2_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_3_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_4_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_5_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_6_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_7_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_8_best.pth.tar",
    "../models/exp213a-deb-l-prompt/fpe_model_fold_9_best.pth.tar",
]


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:41:09.486424Z","iopub.execute_input":"2022-08-23T12:41:09.486846Z","iopub.status.idle":"2022-08-23T12:46:54.870276Z","shell.execute_reply.started":"2022-08-23T12:41:09.486805Z","shell.execute_reply":"2022-08-23T12:46:54.869314Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp213a_dl_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
# del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
gc.collect()
torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:46:54.872785Z","iopub.execute_input":"2022-08-23T12:46:54.87333Z","iopub.status.idle":"2022-08-23T12:46:54.940015Z","shell.execute_reply.started":"2022-08-23T12:46:54.873292Z","shell.execute_reply":"2022-08-23T12:46:54.939247Z"}}
import glob
import pandas as pd

csvs = glob.glob("exp213_dl_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp213_df = pd.DataFrame()
exp213_df["discourse_id"] = idx
exp213_df["Ineffective"] = preds[:, 0]
exp213_df["Adequate"] = preds[:, 1]
exp213_df["Effective"] = preds[:, 2]

exp213_df = exp213_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:46:54.941326Z","iopub.execute_input":"2022-08-23T12:46:54.941831Z","iopub.status.idle":"2022-08-23T12:46:54.953473Z","shell.execute_reply.started":"2022-08-23T12:46:54.941795Z","shell.execute_reply":"2022-08-23T12:46:54.952382Z"}}
exp213_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:46:54.955673Z","iopub.execute_input":"2022-08-23T12:46:54.956367Z","iopub.status.idle":"2022-08-23T12:46:55.016713Z","shell.execute_reply.started":"2022-08-23T12:46:54.956324Z","shell.execute_reply":"2022-08-23T12:46:55.015176Z"}}
import glob
import pandas as pd

csvs = glob.glob("exp213a_dl_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp213a_df = pd.DataFrame()
exp213a_df["discourse_id"] = idx
exp213a_df["Ineffective"] = preds[:, 0]
exp213a_df["Adequate"] = preds[:, 1]
exp213a_df["Effective"] = preds[:, 2]

exp213a_df = exp213a_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:46:55.017973Z","iopub.execute_input":"2022-08-23T12:46:55.018331Z","iopub.status.idle":"2022-08-23T12:46:55.028803Z","shell.execute_reply.started":"2022-08-23T12:46:55.018304Z","shell.execute_reply":"2022-08-23T12:46:55.027839Z"}}
exp213a_df.head()

# %% [markdown]
# #### Full data models

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:46:55.030521Z","iopub.execute_input":"2022-08-23T12:46:55.030893Z","iopub.status.idle":"2022-08-23T12:48:04.68323Z","shell.execute_reply.started":"2022-08-23T12:46:55.030858Z","shell.execute_reply":"2022-08-23T12:48:04.682263Z"}}
if use_full_data_models:

    checkpoints = [
        "../models/exp213f-deb-l-prompt-all/fpe_model_fold_0_best.pth.tar",
        "../models/exp213f-deb-l-prompt-all/fpe_model_fold_1_best.pth.tar",
    ]


    def inference_fn(model, infer_dl, model_id):
        all_preds = []
        all_uids = []
        accelerator = Accelerator()
        model, infer_dl = accelerator.prepare(model, infer_dl)

        model.eval()
        tk0 = tqdm(infer_dl, total=len(infer_dl))

        for batch in tk0:
            with torch.no_grad():
                logits = model(**batch)  # (b, nd, 3)
                batch_preds = F.softmax(logits, dim=-1)
                batch_uids = batch["uids"]
            all_preds.append(batch_preds)
            all_uids.append(batch_uids)

        all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
        all_preds = list(chain(*all_preds))
        flat_preds = list(chain(*all_preds))

        all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
        all_uids = list(chain(*all_uids))
        flat_uids = list(chain(*all_uids))

        preds_df = pd.DataFrame(flat_preds)
        preds_df.columns = ["Ineffective", "Adequate", "Effective"]
        preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
        preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
        preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
        preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
        preds_df.to_csv(f"exp213f_preds_{model_id}.csv", index=False)


    from copy import deepcopy

    for model_id, checkpoint in enumerate(checkpoints):
        # print(f"infering from {checkpoint}")
        new_config = deepcopy(config)

        model = FeedbackModel(new_config)
        # model.half()
        if "swa" in checkpoint:
            ckpt = process_swa_checkpoint(checkpoint)
        else:
            ckpt = torch.load(checkpoint)
            # print(f"validation score for fold {model_id} = {ckpt['loss']}")
        model.load_state_dict(ckpt['state_dict'])
        inference_fn(model, infer_dl, model_id)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:04.685643Z","iopub.execute_input":"2022-08-23T12:48:04.686125Z","iopub.status.idle":"2022-08-23T12:48:04.712681Z","shell.execute_reply.started":"2022-08-23T12:48:04.686081Z","shell.execute_reply":"2022-08-23T12:48:04.711922Z"}}
if use_full_data_models:

    import glob
    import pandas as pd

    csvs = glob.glob("exp213f_preds_*.csv")

    idx = []
    preds = []

    for csv_idx, csv in enumerate(csvs):

        # print("=="*40)
        # print(f"preds in {csv}")
        df = pd.read_csv(csv)
        df = df.sort_values(by=["discourse_id"])
        # print(df.head(10))
        # print("=="*40)

        temp_preds = df.drop(["discourse_id"], axis=1).values
        if csv_idx == 0:
            idx = list(df["discourse_id"])
            preds = temp_preds
        else:
            preds += temp_preds

    preds = preds / len(csvs)

    exp213f_df = pd.DataFrame()
    exp213f_df["discourse_id"] = idx
    exp213f_df["Ineffective"] = preds[:, 0]
    exp213f_df["Adequate"] = preds[:, 1]
    exp213f_df["Effective"] = preds[:, 2]

    exp213f_df = exp213f_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()
    exp213f_df.to_csv("submission.csv", index=False)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:04.715679Z","iopub.execute_input":"2022-08-23T12:48:04.716024Z","iopub.status.idle":"2022-08-23T12:48:04.724925Z","shell.execute_reply.started":"2022-08-23T12:48:04.715992Z","shell.execute_reply":"2022-08-23T12:48:04.724042Z"}}
if use_full_data_models:
    print(exp213f_df.head())

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:04.726477Z","iopub.execute_input":"2022-08-23T12:48:04.726886Z","iopub.status.idle":"2022-08-23T12:48:05.088732Z","shell.execute_reply.started":"2022-08-23T12:48:04.726851Z","shell.execute_reply":"2022-08-23T12:48:05.087786Z"}}
try:
    del tokenizer, infer_dataset, infer_ds_dict, data_collector, infer_dl
    gc.collect()
except Exception as e:
    print(e)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:05.09128Z","iopub.execute_input":"2022-08-23T12:48:05.091722Z","iopub.status.idle":"2022-08-23T12:48:05.480447Z","shell.execute_reply.started":"2022-08-23T12:48:05.091682Z","shell.execute_reply":"2022-08-23T12:48:05.479456Z"}}
try:
    del model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(e)

# %% [markdown]
# # EXP 205 - Deberta-v3-large - MSD + prompt

# %% [markdown]
# ## Config

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:05.48189Z","iopub.execute_input":"2022-08-23T12:48:05.48264Z","iopub.status.idle":"2022-08-23T12:48:05.490464Z","shell.execute_reply.started":"2022-08-23T12:48:05.482608Z","shell.execute_reply":"2022-08-23T12:48:05.489605Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/exp205-debv3-l/mlm_model/",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 8
}
"""
config = json.loads(config)

# %% [markdown]
# ## Dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:05.492365Z","iopub.execute_input":"2022-08-23T12:48:05.492814Z","iopub.status.idle":"2022-08-23T12:48:05.561106Z","shell.execute_reply.started":"2022-08-23T12:48:05.492776Z","shell.execute_reply":"2022-08-23T12:48:05.560223Z"}}
import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION] [COUNTER_CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END] [CLAIM_END]')}")

    print("==" * 40)
    return tokenizer


# --------------- Processing ---------------------------------------------#


DISCOURSE_START_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

TOKEN_MAP = {
    "topic": ["Topic [TOPIC]", "[TOPIC END]"],
    "Lead": ["Lead [LEAD]", "[LEAD END]"],
    "Position": ["Position [POSITION]", "[POSITION END]"],
    "Claim": ["Claim [CLAIM]", "[CLAIM END]"],
    "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM END]"],
    "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL END]"],
    "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE END]"],
    "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT END]"]
}

DISCOURSE_END_TOKENS = [
    "[LEAD END]",
    "[POSITION END]",
    "[CLAIM END]",
    "[COUNTER_CLAIM END]",
    "[REBUTTAL END]",
    "[EVIDENCE END]",
    "[CONCLUDING_STATEMENT END]",
]


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    for cur_discourse in discourse_list:
        if cur_discourse not in to_return:
            to_return[cur_discourse] = []

        matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
        for match in matches:
            span_start, span_end = match.span()
            if span_end <= reading_head:
                continue
            to_return[cur_discourse].append(match.span())
            reading_head = span_end
            break

    # post process
    for cur_discourse in discourse_list:
        if not to_return[cur_discourse]:
            print("resorting to relaxed search...")
            to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, topic, prompt, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    essay_text = "[SOE]" + " [TOPIC] " + prompt + " [TOPIC END] " + essay_text + "[EOE]"
    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")
    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text", "topic", "prompt"]].apply(
        lambda x: process_essay(x[0], x[1], x[2], x[3], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(
            list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


# --------------- Dataset ----------------------------------------------#


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
        }

        self.discourse_type2id = {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)

        # print("adding new tokens...")
        # tokens_to_add = []
        # for this_tok in NEW_TOKENS:
        #     tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
        # self.tokenizer.add_tokens(tokens_to_add)
        print(f"tokenizer len: {len(self.tokenizer)}")

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        self.global_tokens = self.discourse_token_ids.union(self.discourse_end_ids)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset


# --------------- dataset with truncation ---------------------------------------------#


def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

    original_dataset = deepcopy(task_dataset)
    tokenizer = dataset_creator.tokenizer
    START_IDS = dataset_creator.discourse_token_ids
    END_IDS = dataset_creator.discourse_end_ids
    GLOBAL_IDS = dataset_creator.global_tokens

    def tokenize_with_truncation(examples):
        tz = tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=config["max_length"],
            stride=config["stride"],
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )
        return tz

    def process_span(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        buffer = 25  # do not include a head if it is within buffer distance away from last token

        for example_input_ids, example_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            # ------------------- Span Heads -----------------------------------------#
            if len(example_input_ids) < config["max_length"]:  # no truncation
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in START_IDS]
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                        (this_id in START_IDS) & (pos <= config["max_length"] - buffer))]

            n_heads = len(head_candidate)

            # ------------------- Span Tails -----------------------------------------#
            tail_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in END_IDS]

            # ------------------- Edge Cases -----------------------------------------#
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(tail_candidate) > 0) & (len(head_candidate) > 0):
                if tail_candidate[0] < head_candidate[0]:  # truncation effect
                    # print(f"check: heads: {head_candidate}, tails {tail_candidate}")
                    tail_candidate = tail_candidate[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(tail_candidate) < n_heads:
                assert len(tail_candidate) + 1 == n_heads
                assert len(example_input_ids) == config["max_length"]  # should only happen if input text is truncated
                tail_candidate.append(config["max_length"] - 2)  # the token before [SEP] token

            # 3. Additional tails remain in the buffer region
            if len(tail_candidate) > len(head_candidate):
                tail_candidate = tail_candidate[:len(head_candidate)]

            # ------------------- Create the fields ------------------------------------#
            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in head_candidate]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in tail_candidate]

            span_head_idxs.append(head_candidate)
            span_tail_idxs.append(tail_candidate)
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def enforce_alignment(examples):
        uids = []

        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_uids = original_example["uids"]
            char2uid = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_uids)}
            current_example_uids = [char2uid[char_idx] for char_idx in example_span_head_char_start_idxs]
            uids.append(current_example_uids)
        return {"uids": uids}

    def recompute_labels(examples):
        labels = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_labels = original_example["labels"]
            char2label = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_labels)}
            current_example_labels = [char2label[char_idx] for char_idx in example_span_head_char_start_idxs]
            labels.append(current_example_labels)
        return {"labels": labels}

    def recompute_discourse_type_ids(examples):
        discourse_type_ids = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_discourse_type_ids = original_example["discourse_type_ids"]
            char2discourse_id = {k: v for k, v in zip(
                original_example_span_head_char_start_idxs, original_example_discourse_type_ids)}
            current_example_discourse_type_ids = [char2discourse_id[char_idx]
                                                  for char_idx in example_span_head_char_start_idxs]
            discourse_type_ids.append(current_example_discourse_type_ids)
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"head idxs: {head_idxs}, tail idxs {tail_idxs}"

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)
    task_dataset = task_dataset.map(recompute_discourse_type_ids, batched=True)
    task_dataset = task_dataset.map(sanity_check_head_tail, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    if mode != "infer":
        task_dataset = task_dataset.map(recompute_labels, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:05.562314Z","iopub.execute_input":"2022-08-23T12:48:05.563103Z","iopub.status.idle":"2022-08-23T12:48:06.555169Z","shell.execute_reply.started":"2022-08-23T12:48:05.563065Z","shell.execute_reply":"2022-08-23T12:48:06.554248Z"}}
if use_exp205 or use_exp209:
    os.makedirs(config["model_dir"], exist_ok=True)

    print("creating the inference datasets...")
    infer_ds_dict = get_fast_dataset(config, test_df, essay_df, mode="infer")
    tokenizer = infer_ds_dict["tokenizer"]
    infer_dataset = infer_ds_dict["dataset"]
    print(infer_dataset)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.55675Z","iopub.execute_input":"2022-08-23T12:48:06.55713Z","iopub.status.idle":"2022-08-23T12:48:06.573126Z","shell.execute_reply.started":"2022-08-23T12:48:06.557094Z","shell.execute_reply":"2022-08-23T12:48:06.572336Z"}}
if use_exp205 or use_exp209:
    config["len_tokenizer"] = len(tokenizer)

    infer_dataset = infer_dataset.sort("input_length")

    infer_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
                 'span_tail_idxs', 'discourse_type_ids', 'uids']
    )

# %% [markdown]
# ## Data Loader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.575739Z","iopub.execute_input":"2022-08-23T12:48:06.576284Z","iopub.status.idle":"2022-08-23T12:48:06.593019Z","shell.execute_reply.started":"2022-08-23T12:48:06.576247Z","shell.execute_reply":"2022-08-23T12:48:06.592166Z"}}
from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = 512
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in span_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in
                                       discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        # multitask labels
        def _get_additional_labels(label_id):
            if label_id == 0:
                vec = [0, 0]
            elif label_id == 1:
                vec = [1, 0]
            elif label_id == 2:
                vec = [1, 1]
            elif label_id == -1:
                vec = [-1, -1]
            else:
                raise
            return vec

        if labels is not None:
            additional_labels = []
            for ex_labels in batch["labels"]:
                ex_additional_labels = [_get_additional_labels(el) for el in ex_labels]
                additional_labels.append(ex_additional_labels)
            batch["multitask_labels"] = additional_labels
        # pdb.set_trace()

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.594399Z","iopub.execute_input":"2022-08-23T12:48:06.594811Z","iopub.status.idle":"2022-08-23T12:48:06.607083Z","shell.execute_reply.started":"2022-08-23T12:48:06.594776Z","shell.execute_reply":"2022-08-23T12:48:06.606228Z"}}
if use_exp205 or use_exp209:
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    infer_dl = DataLoader(
        infer_dataset,
        batch_size=config["infer_bs"],
        shuffle=False,
        collate_fn=data_collector
    )

# %% [markdown]
# ## Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.608705Z","iopub.execute_input":"2022-08-23T12:48:06.609041Z","iopub.status.idle":"2022-08-23T12:48:06.625724Z","shell.execute_reply.started":"2022-08-23T12:48:06.609014Z","shell.execute_reply":"2022-08-23T12:48:06.624832Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention

from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, DebertaV2Attention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.num_labels = self.config["num_labels"]
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask, **kwargs):

        bs = input_ids.shape[0]  # batch size
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        mean_feature_vector = []
        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)

        return logits




# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.647485Z","iopub.execute_input":"2022-08-23T12:48:06.647747Z","iopub.status.idle":"2022-08-23T12:48:06.659341Z","shell.execute_reply.started":"2022-08-23T12:48:06.647722Z","shell.execute_reply":"2022-08-23T12:48:06.658327Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp205_model_preds_{model_id}.csv", index=False)


if use_exp205:

    for model_id, checkpoint in enumerate(checkpoints):
        print(f"infering from {checkpoint}")
        model = FeedbackModel(config)
        ckpt = torch.load(checkpoint)
        print(f"validation score for fold {model_id} = {ckpt['loss']}")
        model.load_state_dict(ckpt['state_dict'])
        inference_fn(model, infer_dl, model_id)

    del model
    gc.collect()
    torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.66066Z","iopub.execute_input":"2022-08-23T12:48:06.661027Z","iopub.status.idle":"2022-08-23T12:48:06.671975Z","shell.execute_reply.started":"2022-08-23T12:48:06.660993Z","shell.execute_reply":"2022-08-23T12:48:06.671095Z"}}
if use_exp205:
    import glob
    import pandas as pd

    csvs = glob.glob("exp205_model_preds_*.csv")

    idx = []
    preds = []

    for csv_idx, csv in enumerate(csvs):

        print("==" * 40)
        print(f"preds in {csv}")
        df = pd.read_csv(csv)
        df = df.sort_values(by=["discourse_id"])
        print(df.head(10))
        print("==" * 40)

        temp_preds = df.drop(["discourse_id"], axis=1).values
        if csv_idx == 0:
            idx = list(df["discourse_id"])
            preds = temp_preds
        else:
            preds += temp_preds

    preds = preds / len(csvs)

    exp205_df = pd.DataFrame()
    exp205_df["discourse_id"] = idx
    exp205_df["Ineffective"] = preds[:, 0]
    exp205_df["Adequate"] = preds[:, 1]
    exp205_df["Effective"] = preds[:, 2]

    exp205_df = exp205_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T12:48:06.67298Z","iopub.execute_input":"2022-08-23T12:48:06.675963Z","iopub.status.idle":"2022-08-23T12:48:06.685014Z","shell.execute_reply.started":"2022-08-23T12:48:06.675936Z","shell.execute_reply":"2022-08-23T12:48:06.684171Z"}}
if use_exp205:
    exp205_df.head()

# %% [markdown]
# # Exp209 - 10 fold debv3-l

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:06.685816Z","iopub.execute_input":"2022-08-23T12:48:06.686104Z","iopub.status.idle":"2022-08-23T12:48:06.694701Z","shell.execute_reply.started":"2022-08-23T12:48:06.68608Z","shell.execute_reply":"2022-08-23T12:48:06.693822Z"}}
checkpoints = [
    "../models/exp209-debv3-l-prompt/fpe_model_fold_0_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_1_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_2_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_3_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_4_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_5_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_6_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_7_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_8_best.pth.tar",
    "../models/exp209-debv3-l-prompt/fpe_model_fold_9_best.pth.tar",
]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:06.696005Z","iopub.execute_input":"2022-08-23T12:48:06.696688Z","iopub.status.idle":"2022-08-23T12:48:06.713261Z","shell.execute_reply.started":"2022-08-23T12:48:06.696654Z","shell.execute_reply":"2022-08-23T12:48:06.711918Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention

from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, DebertaV2Attention


# -------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """The feedback prize effectiveness baseline model
    """

    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.num_labels = self.config["num_labels"]
        self.classifier = nn.Linear(feature_size, self.config["num_labels"])

    def forward(self, input_ids, attention_mask, span_head_idxs, span_tail_idxs, span_attention_mask, **kwargs):

        bs = input_ids.shape[0]  # batch size
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        mean_feature_vector = []
        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)

        return logits


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:48:06.714743Z","iopub.execute_input":"2022-08-23T12:48:06.715347Z","iopub.status.idle":"2022-08-23T12:54:24.793898Z","shell.execute_reply.started":"2022-08-23T12:48:06.71531Z","shell.execute_reply":"2022-08-23T12:54:24.791949Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp209_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
gc.collect()
torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:54:24.797144Z","iopub.execute_input":"2022-08-23T12:54:24.798625Z","iopub.status.idle":"2022-08-23T12:54:24.86721Z","shell.execute_reply.started":"2022-08-23T12:54:24.798543Z","shell.execute_reply":"2022-08-23T12:54:24.865259Z"}}
import glob
import pandas as pd

csvs = glob.glob("exp209_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp209_df = pd.DataFrame()
exp209_df["discourse_id"] = idx
exp209_df["Ineffective"] = preds[:, 0]
exp209_df["Adequate"] = preds[:, 1]
exp209_df["Effective"] = preds[:, 2]

exp209_df = exp209_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:54:24.868869Z","iopub.execute_input":"2022-08-23T12:54:24.869279Z","iopub.status.idle":"2022-08-23T12:54:24.881069Z","shell.execute_reply.started":"2022-08-23T12:54:24.869241Z","shell.execute_reply":"2022-08-23T12:54:24.880111Z"}}
exp209_df.head()

# %% [markdown]
# #### All data model

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:54:24.882647Z","iopub.execute_input":"2022-08-23T12:54:24.883427Z","iopub.status.idle":"2022-08-23T12:55:04.977367Z","shell.execute_reply.started":"2022-08-23T12:54:24.883389Z","shell.execute_reply":"2022-08-23T12:55:04.976258Z"}}
if use_full_data_models:

    checkpoints = [
        "../models/exp209a-debv3-l-prompt-all/fpe_model_fold_0_best.pth.tar",
    ]


    def inference_fn(model, infer_dl, model_id):
        all_preds = []
        all_uids = []
        accelerator = Accelerator()
        model, infer_dl = accelerator.prepare(model, infer_dl)

        model.eval()
        tk0 = tqdm(infer_dl, total=len(infer_dl))

        for batch in tk0:
            with torch.no_grad():
                logits = model(**batch)  # (b, nd, 3)
                batch_preds = F.softmax(logits, dim=-1)
                batch_uids = batch["uids"]
            all_preds.append(batch_preds)
            all_uids.append(batch_uids)

        all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
        all_preds = list(chain(*all_preds))
        flat_preds = list(chain(*all_preds))

        all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
        all_uids = list(chain(*all_uids))
        flat_uids = list(chain(*all_uids))

        preds_df = pd.DataFrame(flat_preds)
        preds_df.columns = ["Ineffective", "Adequate", "Effective"]
        preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
        preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
        preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
        preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
        preds_df.to_csv(f"exp209a_preds_{model_id}.csv", index=False)


    from copy import deepcopy

    for model_id, checkpoint in enumerate(checkpoints):
        print(f"infering from {checkpoint}")
        new_config = deepcopy(config)
        if "10a" in checkpoint:
            new_config["num_labels"] = 5

        model = FeedbackModel(new_config)
        model.half()
        if "swa" in checkpoint:
            ckpt = process_swa_checkpoint(checkpoint)
        else:
            ckpt = torch.load(checkpoint)
            print(f"validation score for fold {model_id} = {ckpt['loss']}")
        model.load_state_dict(ckpt['state_dict'])
        inference_fn(model, infer_dl, model_id)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:04.97899Z","iopub.execute_input":"2022-08-23T12:55:04.979382Z","iopub.status.idle":"2022-08-23T12:55:05.001963Z","shell.execute_reply.started":"2022-08-23T12:55:04.979346Z","shell.execute_reply":"2022-08-23T12:55:05.001231Z"}}
if use_full_data_models:

    import glob
    import pandas as pd

    csvs = glob.glob("exp209a_preds_*.csv")

    idx = []
    preds = []

    for csv_idx, csv in enumerate(csvs):

        print("==" * 40)
        print(f"preds in {csv}")
        df = pd.read_csv(csv)
        df = df.sort_values(by=["discourse_id"])
        print(df.head(10))
        print("==" * 40)

        temp_preds = df.drop(["discourse_id"], axis=1).values
        if csv_idx == 0:
            idx = list(df["discourse_id"])
            preds = temp_preds
        else:
            preds += temp_preds

    preds = preds / len(csvs)

    exp209a_df = pd.DataFrame()
    exp209a_df["discourse_id"] = idx
    exp209a_df["Ineffective"] = preds[:, 0]
    exp209a_df["Adequate"] = preds[:, 1]
    exp209a_df["Effective"] = preds[:, 2]

    exp209a_df = exp209a_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:05.003153Z","iopub.execute_input":"2022-08-23T12:55:05.003597Z","iopub.status.idle":"2022-08-23T12:55:05.015169Z","shell.execute_reply.started":"2022-08-23T12:55:05.003562Z","shell.execute_reply":"2022-08-23T12:55:05.014227Z"}}
if use_full_data_models:
    print(exp209a_df.head())

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:05.016663Z","iopub.execute_input":"2022-08-23T12:55:05.017708Z","iopub.status.idle":"2022-08-23T12:55:05.416189Z","shell.execute_reply.started":"2022-08-23T12:55:05.017535Z","shell.execute_reply":"2022-08-23T12:55:05.415053Z"}}
try:
    del model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(e)

# %% [markdown]
# # LUKE

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:05.417758Z","iopub.execute_input":"2022-08-23T12:55:05.418216Z","iopub.status.idle":"2022-08-23T12:55:05.426481Z","shell.execute_reply.started":"2022-08-23T12:55:05.418154Z","shell.execute_reply":"2022-08-23T12:55:05.425588Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/luke-span-mlm",
    "model_dir": "./outputs",

    "max_length": 512,
    "max_position_embeddings": 512,
    "stride": 128,
    "max_mention_length": 400,
    "max_entity_length": 24,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 16
}
"""
config = json.loads(config)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:05.429974Z","iopub.execute_input":"2022-08-23T12:55:05.430295Z","iopub.status.idle":"2022-08-23T12:55:05.51474Z","shell.execute_reply.started":"2022-08-23T12:55:05.430264Z","shell.execute_reply":"2022-08-23T12:55:05.513739Z"}}
import os
import pdb
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer, LukeTokenizer


# --------------- Tokenizer ---------------------------------------------#
def tokenizer_test(tokenizer):
    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [==SOE==] [==SPAN==] [==END==]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [==EOE==] [==SPAN==] [==END==]')}")

    print("==" * 40)


def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    return tokenizer


# --------------- Additional Tokens ---------------------------------------------#

TOKEN_MAP = {
    "Lead": ["[==SPAN==]", "[==END==]"],
    "Position": ["[==SPAN==]", "[==END==]"],
    "Claim": ["[==SPAN==]", "[==END==]"],
    "Counterclaim": ["[==SPAN==]", "[==END==]"],
    "Rebuttal": ["[==SPAN==]", "[==END==]"],
    "Evidence": ["[==SPAN==]", "[==END==]"],
    "Concluding Statement": ["[==SPAN==]", "[==END==]"]
}

DISCOURSE_START_TOKENS = [
    "[==SPAN==]",
]

DISCOURSE_END_TOKENS = [
    "[==END==]",
]

NEW_TOKENS = [
    "[==SPAN==]",
    "[==END==]",
    "[==SOE==]",
    "[==EOE==]",
]

ADD_NEW_TOKENS_IN_LUKE = False


# --------------- Data Processing ---------------------------------------------#


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    try:
        for cur_discourse in discourse_list:
            if cur_discourse not in to_return:
                to_return[cur_discourse] = []

            matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
            for match in matches:
                span_start, span_end = match.span()
                if span_end <= reading_head:
                    continue
                to_return[cur_discourse].append(match.span())
                reading_head = span_end
                break

        # post process
        for cur_discourse in discourse_list:
            if not to_return[cur_discourse]:
                print("resorting to relaxed search...")
                to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    except Exception as e:
        pdb.set_trace()
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    essay_text = "[==SOE==]" + essay_text + "[==EOE==]"
    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)
    # pdb.set_trace()
    # set_trace()

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")

    print("--" * 40)
    print("Warning! the following essay_ids are removed during processing...")
    remove_essay_ids = tmp_df[tmp_df["essay_text"].isna()].essay_id.unique()
    print(remove_essay_ids)
    tmp_df = tmp_df[~tmp_df["essay_id"].isin(remove_essay_ids)].copy()
    anno_df = anno_df[~anno_df["essay_id"].isin(remove_essay_ids)].copy()
    notes_df = notes_df[~notes_df["essay_id"].isin(remove_essay_ids)].copy()
    print("--" * 40)

    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text"]].apply(
        lambda x: process_essay(x[0], x[1], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(
            list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


# --------------- Dataset w/o Truncation ----------------------------------------------#


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
            "Mask": -1,
        }

        self.discourse_type2id = {
            "Lead": 0,
            "Position": 1,
            "Claim": 2,
            "Counterclaim": 3,
            "Rebuttal": 4,
            "Evidence": 5,
            "Concluding Statement": 6,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)

        print("adding new tokens...")
        tokens_to_add = []

        for this_tok in NEW_TOKENS:
            tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
        self.tokenizer.add_tokens(tokens_to_add)
        print(f"tokenizer len: {len(self.tokenizer)}")

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))

        tokenizer_test(self.tokenizer)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset


# --------------- Dataset w truncation ---------------------------------------------#


def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

    original_dataset = deepcopy(task_dataset)
    tokenizer = dataset_creator.tokenizer
    START_IDS = dataset_creator.discourse_token_ids
    END_IDS = dataset_creator.discourse_end_ids

    def tokenize_with_truncation(examples):
        tz = tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=config["max_length"],
            stride=config["stride"],
            return_overflowing_tokens=True,
        )
        return tz

    def process_span(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        buffer = 25  # do not include a head if it is within buffer distance away from last token

        for example_input_ids, example_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            # ------------------- Span Heads -----------------------------------------#
            if len(example_input_ids) < config["max_length"]:  # no truncation
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in START_IDS]
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                        (this_id in START_IDS) & (pos <= config["max_length"] - buffer))]

            n_heads = len(head_candidate)

            # ------------------- Span Tails -----------------------------------------#
            tail_candidate = [pos for pos, this_id in enumerate(example_input_ids) if this_id in END_IDS]

            # ------------------- Edge Cases -----------------------------------------#
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(tail_candidate) > 0) & (len(head_candidate) > 0):
                if tail_candidate[0] < head_candidate[0]:  # truncation effect
                    # print(f"check: heads: {head_candidate}, tails {tail_candidate}")
                    tail_candidate = tail_candidate[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(tail_candidate) < n_heads:
                assert len(tail_candidate) + 1 == n_heads
                assert len(example_input_ids) == config["max_length"]  # should only happen if input text is truncated
                tail_candidate.append(config["max_length"] - 2)  # the token before [SEP] token

            # 3. Additional tails remain in the buffer region
            if len(tail_candidate) > len(head_candidate):
                tail_candidate = tail_candidate[:len(head_candidate)]

            # ------------------- Create the fields ------------------------------------#
            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in head_candidate]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in tail_candidate]

            span_head_idxs.append(head_candidate)
            span_tail_idxs.append(tail_candidate)
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def restore_essay_text(examples):
        essay_text = []

        for example_overflow_to_sample_mapping in examples["overflow_to_sample_mapping"]:
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_essay_text = original_example["essay_text"]
            essay_text.append(original_example_essay_text)
        return {"essay_text": essay_text}

    def enforce_alignment(examples):
        uids = []

        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_uids = original_example["uids"]
            char2uid = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_uids)}
            current_example_uids = [char2uid[char_idx] for char_idx in example_span_head_char_start_idxs]
            uids.append(current_example_uids)
        return {"uids": uids}

    def recompute_labels(examples):
        labels = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_labels = original_example["labels"]
            char2label = {k: v for k, v in zip(original_example_span_head_char_start_idxs, original_example_labels)}
            current_example_labels = [char2label[char_idx] for char_idx in example_span_head_char_start_idxs]
            labels.append(current_example_labels)
        return {"labels": labels}

    def recompute_discourse_type_ids(examples):
        discourse_type_ids = []
        for example_span_head_char_start_idxs, example_overflow_to_sample_mapping in zip(
                examples["span_head_char_start_idxs"], examples["overflow_to_sample_mapping"]):
            original_example = original_dataset[example_overflow_to_sample_mapping]
            original_example_span_head_char_start_idxs = original_example["span_head_char_start_idxs"]
            original_example_discourse_type_ids = original_example["discourse_type_ids"]
            char2discourse_id = {k: v for k, v in zip(
                original_example_span_head_char_start_idxs, original_example_discourse_type_ids)}
            current_example_discourse_type_ids = [char2discourse_id[char_idx]
                                                  for char_idx in example_span_head_char_start_idxs]
            discourse_type_ids.append(current_example_discourse_type_ids)
        return {"discourse_type_ids": discourse_type_ids}

    def update_head_tail_char_idx(examples):
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        new_texts = []

        for example_span_head_char_start_idxs, example_span_tail_char_end_idxs, example_offset_mapping, example_essay_text in zip(
                examples["span_head_char_start_idxs"], examples["span_tail_char_end_idxs"], examples["offset_mapping"],
                examples["essay_text"]):
            offset_start = example_offset_mapping[0][0]
            offset_end = example_offset_mapping[-1][1]

            example_essay_text = example_essay_text[offset_start:offset_end]
            new_texts.append(example_essay_text)

            example_span_head_char_start_idxs = [pos - offset_start for pos in example_span_head_char_start_idxs]
            example_span_tail_char_end_idxs = [pos - offset_start for pos in example_span_tail_char_end_idxs]
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)
        return {"span_head_char_start_idxs": span_head_char_start_idxs,
                "span_tail_char_end_idxs": span_tail_char_end_idxs, "essay_text": new_texts}

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"head idxs: {head_idxs}, tail idxs {tail_idxs}"

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)
    task_dataset = task_dataset.map(recompute_discourse_type_ids, batched=True)
    task_dataset = task_dataset.map(sanity_check_head_tail, batched=True)

    task_dataset = task_dataset.map(restore_essay_text, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    if mode != "infer":
        task_dataset = task_dataset.map(recompute_labels, batched=True)

    task_dataset = task_dataset.map(update_head_tail_char_idx, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return


def get_luke_dataset(config, df, essay_df, mode="train"):
    stage_one_config = deepcopy(config)
    stage_one_config["base_model_path"] = "roberta-base"  # Fast Tokenizer
    buffer = 2
    stage_one_config["max_length"] = config["max_length"] - buffer  # - config["max_entity_length"]
    dataset_dict = get_fast_dataset(stage_one_config, df, essay_df, mode)

    task_dataset = dataset_dict["dataset"]

    def get_entity_spans(examples):
        entity_spans = []
        for ex_starts, ex_ends in zip(examples["span_head_char_start_idxs"], examples["span_tail_char_end_idxs"]):
            ex_entity_spans = [tuple([a, b]) for a, b in zip(ex_starts, ex_ends)]
            entity_spans.append(ex_entity_spans)
        return {"entity_spans": entity_spans}

    # prepare luke specific inputs
    task_dataset = task_dataset.map(get_entity_spans, batched=True)

    tokenizer = LukeTokenizer.from_pretrained(
        config["base_model_path"], task="entity_span_classification", max_mention_length=config["max_mention_length"])

    # add new tokens
    if ADD_NEW_TOKENS_IN_LUKE:
        print("adding new tokens...")
        tokens_to_add = []
        for this_tok in NEW_TOKENS:
            tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
        tokenizer.add_tokens(tokens_to_add)

    tokenizer_test(tokenizer)

    def tokenize_with_entity_spans(example):
        tz = tokenizer(
            example["essay_text"],
            entity_spans=[tuple(t) for t in example["entity_spans"]],
            max_entity_length=config["max_entity_length"],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        return tz

    task_dataset = task_dataset.map(tokenize_with_entity_spans, batched=False)

    return_dict = {
        "dataset": task_dataset,
        "tokenizer": tokenizer
    }

    return return_dict

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:05.51664Z","iopub.execute_input":"2022-08-23T12:55:05.517028Z","iopub.status.idle":"2022-08-23T12:55:06.621419Z","shell.execute_reply.started":"2022-08-23T12:55:05.516982Z","shell.execute_reply":"2022-08-23T12:55:06.620471Z"}}
os.makedirs(config["model_dir"], exist_ok=True)

print("creating the inference datasets...")
infer_ds_dict = get_luke_dataset(config, test_df, essay_df, mode="infer")
tokenizer = infer_ds_dict["tokenizer"]
infer_dataset = infer_ds_dict["dataset"]
print(infer_dataset)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:06.62299Z","iopub.execute_input":"2022-08-23T12:55:06.623411Z","iopub.status.idle":"2022-08-23T12:55:06.639034Z","shell.execute_reply.started":"2022-08-23T12:55:06.623374Z","shell.execute_reply":"2022-08-23T12:55:06.638297Z"}}
config["len_tokenizer"] = len(tokenizer)

infer_dataset = infer_dataset.sort("input_length")

infer_dataset.set_format(
    type=None,
    columns=["input_ids", "attention_mask", "entity_ids", "entity_position_ids", "discourse_type_ids",
             "entity_attention_mask", "entity_start_positions", "entity_end_positions", "uids"]
)
#

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:06.640519Z","iopub.execute_input":"2022-08-23T12:55:06.641116Z","iopub.status.idle":"2022-08-23T12:55:06.651871Z","shell.execute_reply.started":"2022-08-23T12:55:06.641073Z","shell.execute_reply":"2022-08-23T12:55:06.651049Z"}}
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in uids])

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_dts + [-1] * (b_max - len(ex_dts)) for ex_dts in discourse_type_ids]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        batch = {k: (torch.tensor(v, dtype=torch.int64)) for k, v in batch.items()}
        return batch


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:06.652884Z","iopub.execute_input":"2022-08-23T12:55:06.655154Z","iopub.status.idle":"2022-08-23T12:55:06.741212Z","shell.execute_reply.started":"2022-08-23T12:55:06.655112Z","shell.execute_reply":"2022-08-23T12:55:06.740248Z"}}
data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

infer_dl = DataLoader(
    infer_dataset,
    batch_size=config["infer_bs"],
    shuffle=False,
    collate_fn=data_collector
)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:06.74534Z","iopub.execute_input":"2022-08-23T12:55:06.746462Z","iopub.status.idle":"2022-08-23T12:55:06.764782Z","shell.execute_reply.started":"2022-08-23T12:55:06.746424Z","shell.execute_reply":"2022-08-23T12:55:06.763999Z"}}
import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertEncoder


# -------- Model ------------------------------------------------------------------#

class FeedbackModel(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update(
            {"max_position_embeddings": config["max_position_embeddings"] + 2}
        )
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        self.num_labels = self.config["num_labels"]

        # LSTM Head
        hidden_size = self.base_model.config.hidden_size

        self.fpe_lstm_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # classification
        feature_size = hidden_size * 3
        self.classifier = nn.Linear(feature_size, self.num_labels)
        self.discourse_classifier = nn.Linear(feature_size, 7)  # 7 discourse elements

        # dropout family
        self.dropout = nn.Dropout(self.config["dropout"])
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

    def forward(
            self,
            input_ids,
            attention_mask,
            entity_ids,
            entity_attention_mask,
            entity_position_ids,
            entity_start_positions,
            entity_end_positions,
            discourse_type_ids,
            **kwargs
    ):
        # get contextual token representations from base transformer
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )

        # run contextual information through lstm
        encoder_layer = outputs.last_hidden_state
        encoder_layer_entity = outputs.entity_last_hidden_state

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        hidden_size = outputs.last_hidden_state.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        start_states = torch.gather(encoder_layer, -2, entity_start_positions)
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        end_states = torch.gather(encoder_layer, -2, entity_end_positions)
        feature_vector = torch.cat([start_states, end_states, encoder_layer_entity],
                                   dim=2)  # check if should use lstm

        feature_vector1 = self.dropout1(feature_vector)
        feature_vector2 = self.dropout2(feature_vector)
        feature_vector3 = self.dropout3(feature_vector)
        feature_vector4 = self.dropout4(feature_vector)
        feature_vector5 = self.dropout5(feature_vector)

        # logits = self.classifier(feature_vector)
        logits1 = self.classifier(feature_vector1)
        logits2 = self.classifier(feature_vector2)
        logits3 = self.classifier(feature_vector3)
        logits4 = self.classifier(feature_vector4)
        logits5 = self.classifier(feature_vector5)
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        return logits


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:55:06.766463Z","iopub.execute_input":"2022-08-23T12:55:06.766974Z","iopub.status.idle":"2022-08-23T12:59:26.355951Z","shell.execute_reply.started":"2022-08-23T12:55:06.766934Z","shell.execute_reply":"2022-08-23T12:59:26.354815Z"}}
checkpoints = [
    "../models/exp17-luke/fpe_model_fold_0_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_1_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_2_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_3_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_4_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_5_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_6_best.pth.tar",
    "../models/exp17-luke/fpe_model_fold_7_best.pth.tar",
]


def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp17_model_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
gc.collect()
torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T12:59:26.358075Z","iopub.execute_input":"2022-08-23T12:59:26.358455Z","iopub.status.idle":"2022-08-23T12:59:26.411388Z","shell.execute_reply.started":"2022-08-23T12:59:26.358419Z","shell.execute_reply":"2022-08-23T12:59:26.410241Z"}}
import glob
import pandas as pd

csvs = glob.glob("exp17_model_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp17_df = pd.DataFrame()
exp17_df["discourse_id"] = idx
exp17_df["Ineffective"] = preds[:, 0]
exp17_df["Adequate"] = preds[:, 1]
exp17_df["Effective"] = preds[:, 2]

exp17_df = exp17_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()
# exp17_df.to_csv("submission.csv", index=False)

exp17_df.head()

# %% [markdown]
# #### LUKE - All data trained

##
checkpoints = [
    "../models/exp17f-luke-all-data-models/fpe_model_fold_0.pth.tar",
    "../models/exp17f-luke-all-data-models/fpe_model_fold_1.pth.tar",
]


def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"exp17f_luke_all_preds_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackModel(config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

####
import glob
import pandas as pd

csvs = glob.glob("exp17f_luke_all_preds_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

exp17f_df = pd.DataFrame()
exp17f_df["discourse_id"] = idx
exp17f_df["Ineffective"] = preds[:, 0]
exp17f_df["Adequate"] = preds[:, 1]
exp17f_df["Effective"] = preds[:, 2]

exp17f_df = exp17f_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()
exp17f_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:00:25.247907Z","iopub.execute_input":"2022-08-23T13:00:25.248301Z","iopub.status.idle":"2022-08-23T13:00:25.650034Z","shell.execute_reply.started":"2022-08-23T13:00:25.248266Z","shell.execute_reply":"2022-08-23T13:00:25.648646Z"}}
del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# # Ensemble

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T13:00:48.848313Z","iopub.execute_input":"2022-08-23T13:00:48.849116Z","iopub.status.idle":"2022-08-23T13:00:48.866003Z","shell.execute_reply.started":"2022-08-23T13:00:48.849078Z","shell.execute_reply":"2022-08-23T13:00:48.865165Z"}}
if use_exp1:
    exp01_df = exp01_df.sort_values(by="discourse_id")


exp19_df = exp19_df.sort_values(by="discourse_id")

exp16_df = exp16_df.sort_values(by="discourse_id")

exp213_df = exp213_df.sort_values(by="discourse_id")
exp213a_df = exp213a_df.sort_values(by="discourse_id")

if use_exp205:
    exp205_df = exp205_df.sort_values(by="discourse_id")

exp209_df = exp209_df.sort_values(by="discourse_id")

if use_exp212:
    exp212_df = exp212_df.sort_values(by="discourse_id")


exp17_df = exp17_df.sort_values(by="discourse_id")

# full data models
exp17f_df = exp17f_df.sort_values(by="discourse_id")  # luke
exp19f_df = exp19f_df.sort_values(by="discourse_id")
exp20f_df = exp20f_df.sort_values(by="discourse_id")
# exp21f_df = exp21f_df.sort_values(by="discourse_id")

exp99_rb_all_df = exp99_rb_all_df.sort_values(by="discourse_id")
exp209a_df = exp209a_df.sort_values(by="discourse_id")
exp213f_df = exp213f_df.sort_values(by="discourse_id")

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:00:50.240295Z","iopub.execute_input":"2022-08-23T13:00:50.241139Z","iopub.status.idle":"2022-08-23T13:00:50.254027Z","shell.execute_reply.started":"2022-08-23T13:00:50.2411Z","shell.execute_reply":"2022-08-23T13:00:50.252937Z"}}
exp17f_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:00:51.208294Z","iopub.execute_input":"2022-08-23T13:00:51.208772Z","iopub.status.idle":"2022-08-23T13:00:51.227323Z","shell.execute_reply.started":"2022-08-23T13:00:51.208729Z","shell.execute_reply":"2022-08-23T13:00:51.22641Z"}}
exp20f_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:21.522716Z","iopub.execute_input":"2022-08-23T13:01:21.523083Z","iopub.status.idle":"2022-08-23T13:01:21.532626Z","shell.execute_reply.started":"2022-08-23T13:01:21.523051Z","shell.execute_reply":"2022-08-23T13:01:21.531485Z"}}
if use_full_data_models:
    print(exp99_rb_all_df.head())

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:22.376912Z","iopub.execute_input":"2022-08-23T13:01:22.377325Z","iopub.status.idle":"2022-08-23T13:01:22.386349Z","shell.execute_reply.started":"2022-08-23T13:01:22.377289Z","shell.execute_reply":"2022-08-23T13:01:22.384254Z"}}
if use_full_data_models:
    print(exp209a_df.head())

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:23.03192Z","iopub.execute_input":"2022-08-23T13:01:23.03421Z","iopub.status.idle":"2022-08-23T13:01:23.043573Z","shell.execute_reply.started":"2022-08-23T13:01:23.034146Z","shell.execute_reply":"2022-08-23T13:01:23.042761Z"}}
if use_full_data_models:
    print(exp213f_df.head())



# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T13:01:25.111888Z","iopub.execute_input":"2022-08-23T13:01:25.112438Z","iopub.status.idle":"2022-08-23T13:01:25.120479Z","shell.execute_reply.started":"2022-08-23T13:01:25.112401Z","shell.execute_reply":"2022-08-23T13:01:25.119531Z"}}
# delv3-mlm40-8folds
if use_exp1:
    print(exp01_df.head())


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T13:01:28.340917Z","iopub.execute_input":"2022-08-23T13:01:28.341593Z","iopub.status.idle":"2022-08-23T13:01:28.351893Z","shell.execute_reply.started":"2022-08-23T13:01:28.341555Z","shell.execute_reply":"2022-08-23T13:01:28.351061Z"}}
# dexl-mlm40-8folds
exp19_df.head()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T13:01:29.824834Z","iopub.execute_input":"2022-08-23T13:01:29.825309Z","iopub.status.idle":"2022-08-23T13:01:29.837549Z","shell.execute_reply.started":"2022-08-23T13:01:29.825272Z","shell.execute_reply":"2022-08-23T13:01:29.836581Z"}}
# deb-l 8 fold mlm40 prompt+ spanfix + msd
exp213_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:30.588016Z","iopub.execute_input":"2022-08-23T13:01:30.58869Z","iopub.status.idle":"2022-08-23T13:01:30.599086Z","shell.execute_reply.started":"2022-08-23T13:01:30.588652Z","shell.execute_reply":"2022-08-23T13:01:30.598222Z"}}
exp213a_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-08-23T13:01:31.223092Z","iopub.execute_input":"2022-08-23T13:01:31.223919Z","iopub.status.idle":"2022-08-23T13:01:31.236239Z","shell.execute_reply.started":"2022-08-23T13:01:31.223883Z","shell.execute_reply":"2022-08-23T13:01:31.235306Z"}}
exp209_df.head()


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:32.972107Z","iopub.execute_input":"2022-08-23T13:01:32.973133Z","iopub.status.idle":"2022-08-23T13:01:32.985238Z","shell.execute_reply.started":"2022-08-23T13:01:32.973097Z","shell.execute_reply":"2022-08-23T13:01:32.98439Z"}}
exp16_df.head()


# %% [markdown]
#  # LSTM

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:36.739239Z","iopub.execute_input":"2022-08-23T13:01:36.740116Z","iopub.status.idle":"2022-08-23T13:01:36.744653Z","shell.execute_reply.started":"2022-08-23T13:01:36.740079Z","shell.execute_reply":"2022-08-23T13:01:36.743718Z"}}
oof_dfs = [
    exp01_df,
    exp19_df,
    exp16_df,
    exp17_df,
    exp209_df,
    exp212_df,
    exp213_df,
    exp213a_df,
]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:37.204629Z","iopub.execute_input":"2022-08-23T13:01:37.205Z","iopub.status.idle":"2022-08-23T13:01:37.212823Z","shell.execute_reply.started":"2022-08-23T13:01:37.204961Z","shell.execute_reply":"2022-08-23T13:01:37.21203Z"}}
pred_cols = ["Ineffective", "Adequate", "Effective"]
for model_idx in range(len(oof_dfs)):
    col_map = dict()
    for col in pred_cols:
        col_map[col] = f"model_{model_idx}_{col}"
    oof_dfs[model_idx] = oof_dfs[model_idx].rename(columns=col_map)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:38.784464Z","iopub.execute_input":"2022-08-23T13:01:38.785048Z","iopub.status.idle":"2022-08-23T13:01:38.817815Z","shell.execute_reply.started":"2022-08-23T13:01:38.78501Z","shell.execute_reply":"2022-08-23T13:01:38.816988Z"}}
merged_df = oof_dfs[0]

for df in oof_dfs[1:]:
    keep_cols = ["discourse_id"] + [col for col in df.columns if col.startswith("model")]
    df = df[keep_cols].copy()
    merged_df = pd.merge(merged_df, df, on="discourse_id", how='inner')
assert merged_df.shape[0] == oof_dfs[0].shape[0]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:39.323082Z","iopub.execute_input":"2022-08-23T13:01:39.323784Z","iopub.status.idle":"2022-08-23T13:01:39.338571Z","shell.execute_reply.started":"2022-08-23T13:01:39.323749Z","shell.execute_reply":"2022-08-23T13:01:39.337605Z"}}
merged_df.head(3).T

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:40.465245Z","iopub.execute_input":"2022-08-23T13:01:40.466221Z","iopub.status.idle":"2022-08-23T13:01:40.473272Z","shell.execute_reply.started":"2022-08-23T13:01:40.466157Z","shell.execute_reply":"2022-08-23T13:01:40.472363Z"}}
feature_names = [col for col in merged_df.columns if col.startswith("model")]
feature_names[:6]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:41.428951Z","iopub.execute_input":"2022-08-23T13:01:41.429637Z","iopub.status.idle":"2022-08-23T13:01:41.436175Z","shell.execute_reply.started":"2022-08-23T13:01:41.4296Z","shell.execute_reply":"2022-08-23T13:01:41.435294Z"}}
feature_map = dict(zip(merged_df["discourse_id"], merged_df[feature_names].values))

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:43.200116Z","iopub.execute_input":"2022-08-23T13:01:43.200548Z","iopub.status.idle":"2022-08-23T13:01:43.264015Z","shell.execute_reply.started":"2022-08-23T13:01:43.200512Z","shell.execute_reply":"2022-08-23T13:01:43.263024Z"}}
import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("==" * 40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION] [COUNTER_CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END] [CLAIM_END]')}")

    print("==" * 40)
    return tokenizer


# --------------- Processing ---------------------------------------------#
USE_NEW_MAP = True

TOKEN_MAP = {
    "Lead": ["Lead [LEAD]", "[LEAD_END]"],
    "Position": ["Position [POSITION]", "[POSITION_END]"],
    "Claim": ["Claim [CLAIM]", "[CLAIM_END]"],
    "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM_END]"],
    "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL_END]"],
    "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE_END]"],
    "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT_END]"]
}

DISCOURSE_START_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

DISCOURSE_END_TOKENS = [
    "[LEAD_END]",
    "[POSITION_END]",
    "[CLAIM_END]",
    "[COUNTER_CLAIM_END]",
    "[REBUTTAL_END]",
    "[EVIDENCE_END]",
    "[CONCLUDING_STATEMENT_END]"
]

if USE_NEW_MAP:
    TOKEN_MAP = {
        "topic": ["Topic [TOPIC]", "[TOPIC END]"],
        "Lead": ["Lead [LEAD]", "[LEAD END]"],
        "Position": ["Position [POSITION]", "[POSITION END]"],
        "Claim": ["Claim [CLAIM]", "[CLAIM END]"],
        "Counterclaim": ["Counterclaim [COUNTER_CLAIM]", "[COUNTER_CLAIM END]"],
        "Rebuttal": ["Rebuttal [REBUTTAL]", "[REBUTTAL END]"],
        "Evidence": ["Evidence [EVIDENCE]", "[EVIDENCE END]"],
        "Concluding Statement": ["Concluding Statement [CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT END]"]
    }

    DISCOURSE_START_TOKENS = [
        "[LEAD]",
        "[POSITION]",
        "[CLAIM]",
        "[COUNTER_CLAIM]",
        "[REBUTTAL]",
        "[EVIDENCE]",
        "[CONCLUDING_STATEMENT]"
    ]

    DISCOURSE_END_TOKENS = [
        "[LEAD END]",
        "[POSITION END]",
        "[CLAIM END]",
        "[COUNTER_CLAIM END]",
        "[REBUTTAL END]",
        "[EVIDENCE END]",
        "[CONCLUDING_STATEMENT END]",
    ]


def relaxed_search(text, substring, min_length=2, fraction=0.99999):
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
            return relaxed_search(text=text,
                                  substring=half_substring,
                                  min_length=min_length,
                                  fraction=fraction)

    span = [position, position + substring_length]
    return span


def build_span_map(discourse_list, essay_text):
    reading_head = 0
    to_return = dict()

    for cur_discourse in discourse_list:
        if cur_discourse not in to_return:
            to_return[cur_discourse] = []

        matches = re.finditer(re.escape(r'{}'.format(cur_discourse)), essay_text)
        for match in matches:
            span_start, span_end = match.span()
            if span_end <= reading_head:
                continue
            to_return[cur_discourse].append(match.span())
            reading_head = span_end
            break

    # post process
    for cur_discourse in discourse_list:
        if not to_return[cur_discourse]:
            print("resorting to relaxed search...")
            to_return[cur_discourse] = [relaxed_search(essay_text, cur_discourse)]
    return to_return


def get_substring_span(texts, mapping):
    result = []
    for text in texts:
        ans = mapping[text].pop(0)
        result.append(ans)
    return result


def process_essay(essay_id, essay_text, anno_df):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    essay_text = "[SOE]" + essay_text + "[EOE]"
    return essay_text


def process_input_df(anno_df, notes_df):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    # ------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")
    tmp_df["span_map"] = tmp_df[["discourse_text", "essay_text"]].apply(
        lambda x: build_span_map(x[0], x[1]), axis=1)
    tmp_df["span"] = tmp_df[["discourse_text", "span_map"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)

    all_discourse_ids = list(chain(*tmp_df["discourse_id"].values))
    all_discourse_spans = list(chain(*tmp_df["span"].values))
    span_df = pd.DataFrame()
    span_df["discourse_id"] = all_discourse_ids
    span_df["span"] = all_discourse_spans
    span_df["discourse_start"] = span_df["span"].apply(lambda x: x[0])
    span_df["discourse_end"] = span_df["span"].apply(lambda x: x[1])
    span_df = span_df.drop(columns="span")

    anno_df = pd.merge(anno_df, span_df, on="discourse_id", how="left")
    # anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text"]].apply(
        lambda x: process_essay(x[0], x[1], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[
            ["uid", "discourse_id", "discourse_effectiveness", "discourse_type"]].agg(list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_id", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids", "discourse_id": "discourse_ids"})

    return grouped_df


# --------------- Dataset ----------------------------------------------#


class FeedbackDatasetMeta:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config

        self.label2id = {
            "Ineffective": 0,
            "Adequate": 1,
            "Effective": 2,
        }

        self.discourse_type2id = {
            "Lead": 1,
            "Position": 2,
            "Claim": 3,
            "Counterclaim": 4,
            "Rebuttal": 5,
            "Evidence": 6,
            "Concluding Statement": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)
        print("==" * 40)
        print("token maps...")
        print(TOKEN_MAP)
        print("==" * 40)
        print(f"tokenizer len: {len(self.tokenizer)}")

        self.discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        self.discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        self.global_tokens = self.discourse_token_ids.union(self.discourse_end_ids)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["essay_text"],
            padding=False,
            truncation=False,  # no truncation at first
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        return tz

    def process_spans(self, examples):

        span_head_char_start_idxs, span_tail_char_end_idxs = [], []
        span_head_idxs, span_tail_idxs = [], []

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"],
                                                                           examples["offset_mapping"],
                                                                           examples["uids"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.discourse_end_ids]

            example_span_head_char_start_idxs = [example_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [example_offset_mapping[pos][1] for pos in example_span_tail_idxs]

            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)

            span_head_idxs.append(example_span_head_idxs)
            span_tail_idxs.append(example_span_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "span_head_char_start_idxs": span_head_char_start_idxs,
            "span_tail_char_end_idxs": span_tail_char_end_idxs,
        }

    def generate_labels(self, examples):
        labels = []
        for example_labels, example_uids in zip(examples["discourse_effectiveness"], examples["uids"]):
            labels.append([self.label2id[l] for l in example_labels])
        return {"labels": labels}

    def generate_meta_features(self, examples):
        meta_features = []
        for example_ids in examples["discourse_ids"]:
            current_features = []
            for didx in example_ids:
                current_features.append(self.feature_map[didx])
            meta_features.append(current_features)
        return {"meta_features": meta_features}

    def generate_discourse_type_ids(self, examples):
        discourse_type_ids = []
        for example_discourse_types in examples["discourse_type"]:
            discourse_type_ids.append([self.discourse_type2id[dt] for dt in example_discourse_types])
        return {"discourse_type_ids": discourse_type_ids}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs)
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1

    def sanity_check_head_labels(self, examples):
        for head_idxs, head_labels in zip(examples["span_head_idxs"], examples["labels"]):
            assert len(head_idxs) == len(head_labels)

    def get_dataset(self, df, essay_df, feature_map, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        self.feature_map = feature_map
        df = process_input_df(df, essay_df)

        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        task_dataset = task_dataset.map(self.generate_meta_features, batched=True)

        print(task_dataset)
        # todo check edge cases
        task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) == len(
            example['span_tail_idxs']))  # no need to run on empty set
        print(task_dataset)
        task_dataset = task_dataset.map(self.generate_discourse_type_ids, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.sanity_check_head_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return task_dataset


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = 512
    return_tensors = "pt"

    def __call__(self, features):
        uids = [feature["uids"] for feature in features]
        discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        meta_features = [feature["meta_features"] for feature in features]

        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        b_max = max([len(l) for l in span_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in
                                       discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        padded_meta_features = []
        for ex_features in meta_features:
            pad_len = b_max - len(ex_features)
            pad_vector = [0. for _ in range(len(ex_features[0]))]
            for _ in range(pad_len):
                ex_features.append(pad_vector)
            padded_meta_features.append(ex_features)
        # set_trace()

        batch["meta_features"] = padded_meta_features

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "meta_features" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:43.49788Z","iopub.execute_input":"2022-08-23T13:01:43.49827Z","iopub.status.idle":"2022-08-23T13:01:43.502741Z","shell.execute_reply.started":"2022-08-23T13:01:43.498235Z","shell.execute_reply":"2022-08-23T13:01:43.501678Z"}}
config = {
    "base_model_path": "../models/exp205-debv3-l/mlm_model",
    "model_dir": "./",
    "valid_bs": 16,
}

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:43.79097Z","iopub.execute_input":"2022-08-23T13:01:43.791756Z","iopub.status.idle":"2022-08-23T13:01:44.491483Z","shell.execute_reply.started":"2022-08-23T13:01:43.791716Z","shell.execute_reply":"2022-08-23T13:01:44.490631Z"}}
dataset_creator = FeedbackDatasetMeta(config)
infer_dataset = dataset_creator.get_dataset(test_df, essay_df, feature_map, mode="infer")

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:44.493054Z","iopub.execute_input":"2022-08-23T13:01:44.493965Z","iopub.status.idle":"2022-08-23T13:01:44.512027Z","shell.execute_reply.started":"2022-08-23T13:01:44.493928Z","shell.execute_reply":"2022-08-23T13:01:44.511229Z"}}
tokenizer = dataset_creator.tokenizer
config["len_tokenizer"] = len(tokenizer)
data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

# sort valid dataset for faster evaluation
infer_dataset = infer_dataset.sort("input_length")

infer_dataset.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
             'span_tail_idxs', 'discourse_type_ids', "meta_features", 'uids']
)

infer_dl = DataLoader(
    infer_dataset,
    batch_size=config["valid_bs"],
    shuffle=False,
    collate_fn=data_collector,
    pin_memory=True,
)


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:44.513629Z","iopub.execute_input":"2022-08-23T13:01:44.514213Z","iopub.status.idle":"2022-08-23T13:01:44.524833Z","shell.execute_reply.started":"2022-08-23T13:01:44.514161Z","shell.execute_reply":"2022-08-23T13:01:44.523988Z"}}
class FeedbackMetaModelResidual(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackMetaModelResidual, self).__init__()

        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]
        # self.layer_norm_raw = LayerNorm(config["num_features"], 1e-7)

        print(f'Num fts: {self.num_meta_features}')
        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, 1e-7)

        self.meta_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(
            self,
            meta_features,
            attention_mask,
            span_attention_mask,
            discourse_type_ids,
            labels=None,
            **kwargs
    ):
        # projection
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # run through rnn
        meta_features_rnn = self.meta_rnn(meta_features)[0]

        # dropout
        meta_features = meta_features + meta_features_rnn
        meta_features = self.dropout(meta_features)
        logits = self.classifier(meta_features)

        return logits


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:44.526818Z","iopub.execute_input":"2022-08-23T13:01:44.527451Z","iopub.status.idle":"2022-08-23T13:01:44.541248Z","shell.execute_reply.started":"2022-08-23T13:01:44.527415Z","shell.execute_reply":"2022-08-23T13:01:44.540061Z"}}
model_config = {
    "num_labels": 3,
    "num_features": len(feature_names),
    "dropout": 0.15,
}
model_config

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:44.650546Z","iopub.execute_input":"2022-08-23T13:01:44.650868Z","iopub.status.idle":"2022-08-23T13:01:44.655455Z","shell.execute_reply.started":"2022-08-23T13:01:44.65084Z","shell.execute_reply":"2022-08-23T13:01:44.654241Z"}}
meta_checkpoints = [
    "../models/ens57-lstm-meta-8m/fpe_model_fold_0_best.pth.tar",
    "../models/ens57-lstm-meta-8m/fpe_model_fold_1_best.pth.tar",
    "../models/ens57-lstm-meta-8m/fpe_model_fold_2_best.pth.tar",
     "../models/ens57-lstm-meta-8m/fpe_model_fold_3_best.pth.tar",
     "../models/ens57-lstm-meta-8m/fpe_model_fold_4_best.pth.tar",
     "../models/ens57-lstm-meta-8m/fpe_model_fold_5_best.pth.tar",
     "../models/ens57-lstm-meta-8m/fpe_model_fold_6_best.pth.tar",
     "../models/ens57-lstm-meta-8m/fpe_model_fold_7_best.pth.tar",
]


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:44.825222Z","iopub.execute_input":"2022-08-23T13:01:44.826054Z","iopub.status.idle":"2022-08-23T13:01:46.659815Z","shell.execute_reply.started":"2022-08-23T13:01:44.826017Z","shell.execute_reply":"2022-08-23T13:01:46.658803Z"}}
def inference_fn(model, infer_dl, model_id):
    all_preds = []
    all_uids = []
    accelerator = Accelerator()
    model, infer_dl = accelerator.prepare(model, infer_dl)

    model.eval()
    tk0 = tqdm(infer_dl, total=len(infer_dl))

    for batch in tk0:
        with torch.no_grad():
            logits = model(**batch)  # (b, nd, 3)
            batch_preds = F.softmax(logits, dim=-1)
            batch_uids = batch["uids"]
        all_preds.append(batch_preds)
        all_uids.append(batch_uids)

    all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
    all_preds = list(chain(*all_preds))
    flat_preds = list(chain(*all_preds))

    all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
    all_uids = list(chain(*all_uids))
    flat_uids = list(chain(*all_uids))

    preds_df = pd.DataFrame(flat_preds)
    preds_df.columns = ["Ineffective", "Adequate", "Effective"]
    preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
    preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
    preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
    preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
    preds_df.to_csv(f"meta_model_{model_id}.csv", index=False)


for model_id, checkpoint in enumerate(meta_checkpoints):
    print(f"infering from {checkpoint}")
    model = FeedbackMetaModelResidual(model_config)
    ckpt = torch.load(checkpoint)
    print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

del model
gc.collect()
torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:46.661998Z","iopub.execute_input":"2022-08-23T13:01:46.662822Z","iopub.status.idle":"2022-08-23T13:01:46.718614Z","shell.execute_reply.started":"2022-08-23T13:01:46.662783Z","shell.execute_reply":"2022-08-23T13:01:46.716253Z"}}
import glob
import pandas as pd

csvs = glob.glob("meta_model_*.csv")

idx = []
preds = []

for csv_idx, csv in enumerate(csvs):

    print("==" * 40)
    print(f"preds in {csv}")
    df = pd.read_csv(csv)
    df = df.sort_values(by=["discourse_id"])
    print(df.head(10))
    print("==" * 40)

    temp_preds = df.drop(["discourse_id"], axis=1).values
    if csv_idx == 0:
        idx = list(df["discourse_id"])
        preds = temp_preds
    else:
        preds += temp_preds

preds = preds / len(csvs)

meta_pred_df = pd.DataFrame()
meta_pred_df["discourse_id"] = idx
meta_pred_df["Ineffective"] = preds[:, 0]
meta_pred_df["Adequate"] = preds[:, 1]
meta_pred_df["Effective"] = preds[:, 2]

meta_pred_df = meta_pred_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:46.719777Z","iopub.execute_input":"2022-08-23T13:01:46.72025Z","iopub.status.idle":"2022-08-23T13:01:46.731523Z","shell.execute_reply.started":"2022-08-23T13:01:46.720216Z","shell.execute_reply":"2022-08-23T13:01:46.730725Z"}}
meta_pred_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:46.733571Z","iopub.execute_input":"2022-08-23T13:01:46.734176Z","iopub.status.idle":"2022-08-23T13:01:46.747592Z","shell.execute_reply.started":"2022-08-23T13:01:46.734138Z","shell.execute_reply":"2022-08-23T13:01:46.746696Z"}}
submission_df = meta_pred_df.copy()  # pd.DataFrame()
submission_df.head(10)

# %% [markdown]
# ## LGB

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:46.749896Z","iopub.execute_input":"2022-08-23T13:01:46.750557Z","iopub.status.idle":"2022-08-23T13:01:46.777264Z","shell.execute_reply.started":"2022-08-23T13:01:46.750522Z","shell.execute_reply":"2022-08-23T13:01:46.776507Z"}}
merged_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:46.778806Z","iopub.execute_input":"2022-08-23T13:01:46.779355Z","iopub.status.idle":"2022-08-23T13:01:46.807299Z","shell.execute_reply.started":"2022-08-23T13:01:46.779321Z","shell.execute_reply":"2022-08-23T13:01:46.806297Z"}}
from textblob import TextBlob

meta_df = pd.merge(merged_df, test_df, on="discourse_id", how="left")
meta_df = pd.merge(meta_df, essay_df, on="essay_id", how="left")


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

    span = [position, position + substring_length]
    return span


def tags(text):
    blob = TextBlob(text)
    return blob


def count_typ_tags(text, typ):
    return len([word for (word, tag) in text.tags if tag.startswith(typ)])


def get_features(meta_df):
    config = dict()
    feature_names = [col for col in merged_df.columns if col.startswith("model")]

    config["features"] = feature_names
    config["cat_features"] = []

    print('Processing spans')
    meta_df["discourse_span"] = meta_df[["essay_text", "discourse_text"]].apply(
        lambda x: get_substring_span(x[0], x[1]), axis=1)
    meta_df["discourse_start"] = meta_df["discourse_span"].apply(lambda x: x[0])
    meta_df["discourse_end"] = meta_df["discourse_span"].apply(lambda x: x[1])

    meta_df['discourse_len'] = meta_df['discourse_end'] - meta_df['discourse_start']
    meta_df['freq_of_essay_id'] = meta_df['essay_id'].map(dict(meta_df['essay_id'].value_counts()))
    meta_df['blob_discourse'] = meta_df['discourse_text'].apply(tags)
    meta_df['discourse_Adjectives'] = meta_df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'JJ'))
    meta_df['discourse_Verbs'] = meta_df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'VB'))
    meta_df['discourse_Adverbs'] = meta_df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'RB'))
    meta_df['discourse_Nouns'] = meta_df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'NN'))
    meta_df['discourse_VBP'] = meta_df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'VBP'))
    meta_df['discourse_PRP'] = meta_df['blob_discourse'].apply(lambda x: count_typ_tags(x, 'PRP'))
    meta_df['count_next_line_essay'] = meta_df['essay_text'].apply(lambda x: x.count("\n\n"))

    discourse_type2id = {
        "Lead": 1,
        "Position": 2,
        "Claim": 3,
        "Counterclaim": 4,
        "Rebuttal": 5,
        "Evidence": 6,
        "Concluding Statement": 7,
    }

    new_col = []
    for unique in ['Claim']:
        meta_df['is_' + unique] = meta_df['discourse_type'].apply(lambda x: 1 if x == unique else 0)
        new_col.append('is_' + unique)

    meta_df = meta_df.sort_values(by=['essay_id', 'discourse_id']).reset_index(drop=True)

    essay_discourse_list = meta_df.groupby(['essay_id']).apply( \
        lambda x: x['discourse_type'].tolist()).reset_index()
    essay_discourse_list.rename(columns={0: 'discourse_type_list'}, inplace=True)
    essay_discourse_list['discourse_type_list'] = essay_discourse_list['discourse_type_list'].apply(
        lambda x: " ".join(x))
    meta_df = meta_df.merge(essay_discourse_list[['essay_id', 'discourse_type_list']], \
                            on='essay_id', how='left')
    meta_df["discourse_type"] = meta_df["discourse_type"].map(discourse_type2id)
    meta_df['discourse_type_fe'] = meta_df['discourse_type'].map(dict(meta_df['discourse_type'].value_counts()))

    essay_discourse = meta_df.groupby(['essay_id']).apply(lambda x: \
                                                              x['discourse_type'].nunique()).reset_index()
    essay_discourse.rename(columns={0: 'unique_discourse_type'}, inplace=True)
    essay_discourse.head()

    meta_df = meta_df.merge(essay_discourse[['essay_id', 'unique_discourse_type']], on='essay_id', \
                            how='left')

    config["features"].extend(["discourse_type",
                               "discourse_type_fe", "discourse_len", "freq_of_essay_id", \
                               "unique_discourse_type"] + new_col)
    config['features'].extend(['discourse_Adjectives', 'discourse_Verbs', \
                               'discourse_Adverbs', 'discourse_Nouns', \
                               'count_next_line_essay', 'discourse_VBP', 'discourse_PRP'])
    config["cat_features"].append("discourse_type")

    return meta_df, config


# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:46.808644Z","iopub.execute_input":"2022-08-23T13:01:46.809636Z","iopub.status.idle":"2022-08-23T13:01:47.031822Z","shell.execute_reply.started":"2022-08-23T13:01:46.809552Z","shell.execute_reply":"2022-08-23T13:01:47.031022Z"}}
meta_df, config = get_features(meta_df)
meta_df.shape, len(config['features'])

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:47.033319Z","iopub.execute_input":"2022-08-23T13:01:47.033851Z","iopub.status.idle":"2022-08-23T13:01:47.416904Z","shell.execute_reply.started":"2022-08-23T13:01:47.033813Z","shell.execute_reply":"2022-08-23T13:01:47.415996Z"}}
import lightgbm as lgbm
import joblib

model_paths = [
    "../models/meta-lgbm-model/lgbm_model_fold_0.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_1.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_2.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_3.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_4.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_5.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_6.txt",
    "../models/meta-lgbm-model/lgbm_model_fold_7.txt",
]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:47.418284Z","iopub.execute_input":"2022-08-23T13:01:47.418741Z","iopub.status.idle":"2022-08-23T13:01:48.343755Z","shell.execute_reply.started":"2022-08-23T13:01:47.418701Z","shell.execute_reply":"2022-08-23T13:01:48.342691Z"}}
for midx, mp in enumerate(model_paths):
    model = lgbm.Booster(model_file=mp)
    if midx == 0:
        preds = model.predict(meta_df[config["features"]], num_iteration=model.best_iteration)
    else:
        preds += model.predict(meta_df[config["features"]], num_iteration=model.best_iteration)
preds = preds / len(model_paths)
preds
submission_df1 = pd.DataFrame()

submission_df1["discourse_id"] = meta_df["discourse_id"].values
submission_df1["Ineffective"] = preds[:, 0]
submission_df1["Adequate"] = preds[:, 1]
submission_df1["Effective"] = preds[:, 2]

# %% [markdown]
# # All Data

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.348096Z","iopub.execute_input":"2022-08-23T13:01:48.34857Z","iopub.status.idle":"2022-08-23T13:01:48.395638Z","shell.execute_reply.started":"2022-08-23T13:01:48.348529Z","shell.execute_reply":"2022-08-23T13:01:48.394659Z"}}
df_list = [
    exp17f_df,
    exp19f_df,
    exp20f_df,
    exp99_rb_all_df,
    exp209a_df,
    exp213f_df,

]

MODEL_WEIGHTS = [
    0.08,  # [2] luke -rb
    0.18,  # [2] dexl -rb
    0.04,  # [1] del kd -rb
    0.15,  # [2] delv3 -rb
    0.20,  # [1] delv3 - tk
    0.35,  # [2] del - tk
]

print(f"sum of weights {np.sum(MODEL_WEIGHTS)}")

all_data_df = pd.DataFrame()
all_data_df["discourse_id"] = df_list[0]["discourse_id"].values

for model_idx, model_preds in enumerate(df_list):
    if model_idx == 0:
        all_data_df["Ineffective"] = MODEL_WEIGHTS[model_idx] * model_preds["Ineffective"]
        all_data_df["Adequate"] = MODEL_WEIGHTS[model_idx] * model_preds["Adequate"]
        all_data_df["Effective"] = MODEL_WEIGHTS[model_idx] * model_preds["Effective"]
    else:
        all_data_df["Ineffective"] += MODEL_WEIGHTS[model_idx] * model_preds["Ineffective"]
        all_data_df["Adequate"] += MODEL_WEIGHTS[model_idx] * model_preds["Adequate"]
        all_data_df["Effective"] += MODEL_WEIGHTS[model_idx] * model_preds["Effective"]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.397526Z","iopub.execute_input":"2022-08-23T13:01:48.398179Z","iopub.status.idle":"2022-08-23T13:01:48.418433Z","shell.execute_reply.started":"2022-08-23T13:01:48.398137Z","shell.execute_reply":"2022-08-23T13:01:48.41691Z"}}
all_data_df.head()

# %% [markdown]
# # Final Ensemble

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.422626Z","iopub.execute_input":"2022-08-23T13:01:48.423236Z","iopub.status.idle":"2022-08-23T13:01:48.434273Z","shell.execute_reply.started":"2022-08-23T13:01:48.423176Z","shell.execute_reply":"2022-08-23T13:01:48.433357Z"}}
lgb_df = submission_df1.sort_values(by="discourse_id")
lstm_df = submission_df.sort_values(by="discourse_id")
all_data_df = all_data_df.sort_values(by="discourse_id")  # TODO: check for flag

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.438253Z","iopub.execute_input":"2022-08-23T13:01:48.438582Z","iopub.status.idle":"2022-08-23T13:01:48.456725Z","shell.execute_reply.started":"2022-08-23T13:01:48.43855Z","shell.execute_reply":"2022-08-23T13:01:48.455838Z"}}
lgb_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.460469Z","iopub.execute_input":"2022-08-23T13:01:48.462833Z","iopub.status.idle":"2022-08-23T13:01:48.479883Z","shell.execute_reply.started":"2022-08-23T13:01:48.462799Z","shell.execute_reply":"2022-08-23T13:01:48.479193Z"}}
lstm_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.585816Z","iopub.execute_input":"2022-08-23T13:01:48.586258Z","iopub.status.idle":"2022-08-23T13:01:48.604969Z","shell.execute_reply.started":"2022-08-23T13:01:48.586217Z","shell.execute_reply":"2022-08-23T13:01:48.604303Z"}}
all_data_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.787491Z","iopub.execute_input":"2022-08-23T13:01:48.788285Z","iopub.status.idle":"2022-08-23T13:01:48.793903Z","shell.execute_reply.started":"2022-08-23T13:01:48.788242Z","shell.execute_reply":"2022-08-23T13:01:48.792869Z"}}
# 0.536321*0.25 + 0.616211*0.35 + 0.499419*0.4
# exp99_rb_all_df.head(1)
# exp209a_df.head(1)
# exp213f_df.head(1)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:48.97001Z","iopub.execute_input":"2022-08-23T13:01:48.970384Z","iopub.status.idle":"2022-08-23T13:01:48.985054Z","shell.execute_reply.started":"2022-08-23T13:01:48.970351Z","shell.execute_reply":"2022-08-23T13:01:48.984236Z"}}
sub_df = pd.DataFrame()
sub_df["discourse_id"] = lgb_df["discourse_id"].values

lgb_vals = lgb_df[["Ineffective", "Adequate", "Effective"]].values
lstm_vals = lstm_df[["Ineffective", "Adequate", "Effective"]].values
all_vals = all_data_df[["Ineffective", "Adequate", "Effective"]].values

sub_df["Ineffective"] = 0.60 * (0.7 * lstm_vals[:, 0] + 0.3 * lgb_vals[:, 0]) + 0.40 * all_vals[:, 0]
sub_df["Adequate"] = 0.60 * (0.7 * lstm_vals[:, 1] + 0.3 * lgb_vals[:, 1]) + 0.40 * all_vals[:, 1]
sub_df["Effective"] = 0.60 * (0.7 * lstm_vals[:, 2] + 0.3 * lgb_vals[:, 2]) + 0.40 * all_vals[:, 2]

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:49.180054Z","iopub.execute_input":"2022-08-23T13:01:49.180994Z","iopub.status.idle":"2022-08-23T13:01:49.199886Z","shell.execute_reply.started":"2022-08-23T13:01:49.180949Z","shell.execute_reply":"2022-08-23T13:01:49.198711Z"}}
sub_df.to_csv("submission.csv", index=False)
sub_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-08-23T13:01:49.39693Z","iopub.execute_input":"2022-08-23T13:01:49.397308Z","iopub.status.idle":"2022-08-23T13:01:49.411781Z","shell.execute_reply.started":"2022-08-23T13:01:49.397275Z","shell.execute_reply":"2022-08-23T13:01:49.410813Z"}}
sub_df.to_csv("submission.csv", index=False)
sub_df.head()

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
