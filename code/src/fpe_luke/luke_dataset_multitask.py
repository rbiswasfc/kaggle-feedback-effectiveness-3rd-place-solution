import os
import pdb
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer, LukeTokenizer


#--------------- Tokenizer ---------------------------------------------#
def tokenizer_test(tokenizer):
    print("=="*40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [==SOE==] [==SPAN==] [==END==]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [==EOE==] [==SPAN==] [==END==]')}")

    print("=="*40)


def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    return tokenizer


#--------------- Additional Tokens ---------------------------------------------#

TOKEN_MAP = {
    "Lead":                     ["[==SPAN==]", "[==END==]"],
    "Position":                 ["[==SPAN==]", "[==END==]"],
    "Claim":                    ["[==SPAN==]", "[==END==]"],
    "Counterclaim":             ["[==SPAN==]", "[==END==]"],
    "Rebuttal":                 ["[==SPAN==]", "[==END==]"],
    "Evidence":                 ["[==SPAN==]", "[==END==]"],
    "Concluding Statement":     ["[==SPAN==]", "[==END==]"]
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
#--------------- Data Processing ---------------------------------------------#


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

    span = [position, position+substring_length]
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

    #------------------- Pre-Process Essay Text --------------------------#
    anno_df["discourse_text"] = anno_df["discourse_text"].apply(lambda x: x.strip())  # pre-process
    if "discourse_effectiveness" in anno_df.columns:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text",
                           "discourse_type", "discourse_effectiveness", "uid"]].copy()
    else:
        anno_df = anno_df[["discourse_id", "essay_id", "discourse_text", "discourse_type", "uid"]].copy()

    tmp_df = anno_df.groupby("essay_id")[["discourse_id", "discourse_text"]].agg(list).reset_index()
    tmp_df = pd.merge(tmp_df, notes_df, on="essay_id", how="left")

    print("--"*40)
    print("Warning! the following essay_ids are removed during processing...")
    remove_essay_ids = tmp_df[tmp_df["essay_text"].isna()].essay_id.unique()
    print(remove_essay_ids)
    tmp_df = tmp_df[~tmp_df["essay_id"].isin(remove_essay_ids)].copy()
    anno_df = anno_df[~anno_df["essay_id"].isin(remove_essay_ids)].copy()
    notes_df = notes_df[~notes_df["essay_id"].isin(remove_essay_ids)].copy()
    print("--"*40)

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

    print("=="*40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text"]].apply(
        lambda x: process_essay(x[0], x[1], anno_df), axis=1
    )
    print("=="*40)

    anno_df = anno_df.drop(columns=["discourse_start", "discourse_end"])
    notes_df = notes_df.drop_duplicates(subset=["essay_id"])[["essay_id", "essay_text"]].copy()

    anno_df = pd.merge(anno_df, notes_df, on="essay_id", how="left")

    if "discourse_effectiveness" in anno_df.columns:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_effectiveness", "discourse_type"]].agg(list).reset_index()
    else:
        grouped_df = anno_df.groupby("essay_id")[["uid", "discourse_type"]].agg(list).reset_index()

    grouped_df = pd.merge(grouped_df, notes_df, on="essay_id", how="left")
    grouped_df = grouped_df.rename(columns={"uid": "uids"})

    return grouped_df


#--------------- Dataset w/o Truncation ----------------------------------------------#


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
        print("=="*40)
        print("token maps...")
        print(TOKEN_MAP)
        print("=="*40)

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

        for example_input_ids, example_offset_mapping, example_uids in zip(examples["input_ids"], examples["offset_mapping"], examples["uids"]):
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

#--------------- Dataset w truncation ---------------------------------------------#


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
                    (this_id in START_IDS) & (pos <= config["max_length"]-buffer))]

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
                tail_candidate.append(config["max_length"]-2)  # the token before [SEP] token

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
                examples["span_head_char_start_idxs"], examples["span_tail_char_end_idxs"], examples["offset_mapping"], examples["essay_text"]):

            offset_start = example_offset_mapping[0][0]
            offset_end = example_offset_mapping[-1][1]

            example_essay_text = example_essay_text[offset_start:offset_end]
            new_texts.append(example_essay_text)

            example_span_head_char_start_idxs = [pos - offset_start for pos in example_span_head_char_start_idxs]
            example_span_tail_char_end_idxs = [pos - offset_start for pos in example_span_tail_char_end_idxs]
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)
        return {"span_head_char_start_idxs": span_head_char_start_idxs, "span_tail_char_end_idxs": span_tail_char_end_idxs, "essay_text": new_texts}

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
