# basics
import os
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
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

# misc
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ipython
from IPython.display import display
from IPython.core.debugger import set_trace

# %% [markdown]
# # Config

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:32:19.307976Z","iopub.execute_input":"2022-08-18T16:32:19.308767Z","iopub.status.idle":"2022-08-18T16:32:19.317331Z","shell.execute_reply.started":"2022-08-18T16:32:19.308716Z","shell.execute_reply":"2022-08-18T16:32:19.316269Z"}}
config = """{
    "debug": false,

    "base_model_path": "../models/deberta-large-prompt-mlm40",
    "model_dir": "./outputs",

    "max_length": 1024,
    "stride": 256,
    "num_labels": 3,
    "dropout": 0.1,
    "infer_bs": 16
}
"""
config = json.loads(config)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:32:19.319459Z","iopub.execute_input":"2022-08-18T16:32:19.319878Z","iopub.status.idle":"2022-08-18T16:33:13.419696Z","shell.execute_reply.started":"2022-08-18T16:32:19.319843Z","shell.execute_reply":"2022-08-18T16:33:13.418493Z"}}
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

# %% [markdown]
# # Load Data

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:13.421005Z","iopub.execute_input":"2022-08-18T16:33:13.421357Z","iopub.status.idle":"2022-08-18T16:33:15.761865Z","shell.execute_reply.started":"2022-08-18T16:33:13.42132Z","shell.execute_reply":"2022-08-18T16:33:15.755882Z"}}
test_df = pd.read_csv("../datasets/feedback-prize-effectiveness/test.csv")
all_ids = test_df["discourse_id"].unique().tolist()
discourse2idx = {discourse: pos for pos, discourse in enumerate(all_ids)}
idx2discourse = {v: k for k, v in discourse2idx.items()}
test_df["uid"] = test_df["discourse_id"].map(discourse2idx)


def _load_essay(essay_id):
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
# display(test_df.sample())

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:15.78196Z","iopub.execute_input":"2022-08-18T16:33:15.788049Z","iopub.status.idle":"2022-08-18T16:33:15.823436Z","shell.execute_reply.started":"2022-08-18T16:33:15.787996Z","shell.execute_reply":"2022-08-18T16:33:15.817671Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:15.832694Z","iopub.execute_input":"2022-08-18T16:33:15.833404Z","iopub.status.idle":"2022-08-18T16:33:15.879831Z","shell.execute_reply.started":"2022-08-18T16:33:15.833368Z","shell.execute_reply":"2022-08-18T16:33:15.874816Z"}}
essay_df = essay_df.merge(pred_topics, on='essay_id', how='left')
essay_df['prompt'] = essay_df['topic'].map(topic_map)

# essay_df.head()

# %% [markdown]
# # Dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:15.884127Z","iopub.execute_input":"2022-08-18T16:33:15.887997Z","iopub.status.idle":"2022-08-18T16:33:16.265044Z","shell.execute_reply.started":"2022-08-18T16:33:15.887957Z","shell.execute_reply":"2022-08-18T16:33:16.263924Z"}}
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


def process_essay(essay_id, essay_text, topic, anno_df):
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

    essay_text = "[SOE]" + " [TOPIC] " + str(topic) + " [TOPIC END] " + essay_text + "[EOE]"
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

    print("==" * 40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text", "prompt"]].apply(
        lambda x: process_essay(x[0], x[1], x[2], anno_df), axis=1
    )
    print("==" * 40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")
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

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return task_dataset


# --------------- dataset with truncation ---------------------------------------------#

def get_fast_dataset(config, df, essay_df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    task_dataset = dataset_creator.get_dataset(df, essay_df, mode=mode)

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

    def compute_input_length(examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    task_dataset = task_dataset.map(
        tokenize_with_truncation,
        batched=True,
        remove_columns=task_dataset.column_names,
        batch_size=len(task_dataset)
    )

    task_dataset = task_dataset.map(process_span, batched=True)
    task_dataset = task_dataset.map(enforce_alignment, batched=True)

    # no need to run on empty set
    task_dataset = task_dataset.filter(lambda example: len(example['span_head_idxs']) != 0)
    task_dataset = task_dataset.map(compute_input_length, batched=True)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["original_dataset"] = original_dataset
    to_return["tokenizer"] = tokenizer
    return to_return

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:16.266804Z","iopub.execute_input":"2022-08-18T16:33:16.2675Z","iopub.status.idle":"2022-08-18T16:33:17.107288Z","shell.execute_reply.started":"2022-08-18T16:33:16.267463Z","shell.execute_reply":"2022-08-18T16:33:17.106301Z"}}
os.makedirs(config["model_dir"], exist_ok=True)

print("creating the inference datasets...")
infer_ds_dict = get_fast_dataset(config, test_df, essay_df, mode="infer")
tokenizer = infer_ds_dict["tokenizer"]
infer_dataset = infer_ds_dict["dataset"]
# print(infer_dataset)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:17.10874Z","iopub.execute_input":"2022-08-18T16:33:17.109807Z","iopub.status.idle":"2022-08-18T16:33:17.126715Z","shell.execute_reply.started":"2022-08-18T16:33:17.109766Z","shell.execute_reply":"2022-08-18T16:33:17.125873Z"}}
config["len_tokenizer"] = len(tokenizer)

infer_dataset = infer_dataset.sort("input_length")

infer_dataset.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
             'span_tail_idxs', 'uids']
)

# %% [markdown]
# # DataLoader

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:17.128155Z","iopub.execute_input":"2022-08-18T16:33:17.128493Z","iopub.status.idle":"2022-08-18T16:33:17.144347Z","shell.execute_reply.started":"2022-08-18T16:33:17.128459Z","shell.execute_reply":"2022-08-18T16:33:17.143082Z"}}
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
        #         discourse_type_ids = [feature["discourse_type_ids"] for feature in features]
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
        #         batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
        #                                        (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

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


# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:17.146045Z","iopub.execute_input":"2022-08-18T16:33:17.146892Z","iopub.status.idle":"2022-08-18T16:33:17.158351Z","shell.execute_reply.started":"2022-08-18T16:33:17.146738Z","shell.execute_reply":"2022-08-18T16:33:17.157266Z"}}
data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

infer_dl = DataLoader(
    infer_dataset,
    batch_size=config["infer_bs"],
    shuffle=False,
    collate_fn=data_collector
)

# %% [markdown]
# # Model

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:17.159843Z","iopub.execute_input":"2022-08-18T16:33:17.160389Z","iopub.status.idle":"2022-08-18T16:33:17.177613Z","shell.execute_reply.started":"2022-08-18T16:33:17.160351Z","shell.execute_reply":"2022-08-18T16:33:17.176465Z"}}
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
# # Inference

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:17.178941Z","iopub.execute_input":"2022-08-18T16:33:17.179376Z","iopub.status.idle":"2022-08-18T16:33:17.19203Z","shell.execute_reply.started":"2022-08-18T16:33:17.179334Z","shell.execute_reply":"2022-08-18T16:33:17.190921Z"}}
def process_swa_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path)

    print("processing ckpt...")
    print("removing module from keys...")
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k == "n_averaged":
            print(f"# of snapshots in {checkpoint_path} = {v}")
            continue
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    processed_state = {"state_dict": new_state_dict}

    # delete old state
    del state_dict
    gc.collect()

    return processed_state


# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:17.196683Z","iopub.execute_input":"2022-08-18T16:33:17.197136Z","iopub.status.idle":"2022-08-18T16:33:52.644275Z","shell.execute_reply.started":"2022-08-18T16:33:17.197058Z","shell.execute_reply":"2022-08-18T16:33:52.642595Z"}}
checkpoints = [
    "../models/exp213f-deb-l-prompt-all/fpe_model_fold_0_best.pth.tar",
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
    preds_df.to_csv(f"preds_{model_id}.csv", index=False)


from copy import deepcopy

for model_id, checkpoint in enumerate(checkpoints):
    # print(f"infering from {checkpoint}")
    new_config = deepcopy(config)
    if "10a" in checkpoint:
        new_config["num_labels"] = 5

    model = FeedbackModel(new_config)
    # model.half()
    if "swa" in checkpoint:
        ckpt = process_swa_checkpoint(checkpoint)
    else:
        ckpt = torch.load(checkpoint)
        # print(f"validation score for fold {model_id} = {ckpt['loss']}")
    model.load_state_dict(ckpt['state_dict'])
    inference_fn(model, infer_dl, model_id)

# %% [markdown]
# # Ensemble

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:52.645808Z","iopub.execute_input":"2022-08-18T16:33:52.646164Z","iopub.status.idle":"2022-08-18T16:33:52.668707Z","shell.execute_reply.started":"2022-08-18T16:33:52.646129Z","shell.execute_reply":"2022-08-18T16:33:52.66787Z"}}
import glob
import pandas as pd

csvs = glob.glob("preds_*.csv")

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

submission_df = pd.DataFrame()
submission_df["discourse_id"] = idx
submission_df["Ineffective"] = preds[:, 0]
submission_df["Adequate"] = preds[:, 1]
submission_df["Effective"] = preds[:, 2]

submission_df = submission_df.groupby("discourse_id")[["Ineffective", "Adequate", "Effective"]].agg(
    np.mean).reset_index()
submission_df.to_csv("submission.csv", index=False)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:57.305333Z","iopub.execute_input":"2022-08-18T16:33:57.306247Z","iopub.status.idle":"2022-08-18T16:33:57.324477Z","shell.execute_reply.started":"2022-08-18T16:33:57.30619Z","shell.execute_reply":"2022-08-18T16:33:57.323427Z"}}
# submission_df.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:52.676674Z","iopub.execute_input":"2022-08-18T16:33:52.677371Z","iopub.status.idle":"2022-08-18T16:33:52.685294Z","shell.execute_reply.started":"2022-08-18T16:33:52.677336Z","shell.execute_reply":"2022-08-18T16:33:52.684513Z"}}
# submission_df.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-08-18T16:33:52.688154Z","iopub.execute_input":"2022-08-18T16:33:52.688704Z","iopub.status.idle":"2022-08-18T16:33:52.695453Z","shell.execute_reply.started":"2022-08-18T16:33:52.688677Z","shell.execute_reply":"2022-08-18T16:33:52.694594Z"}}
##### --- End ---------#####