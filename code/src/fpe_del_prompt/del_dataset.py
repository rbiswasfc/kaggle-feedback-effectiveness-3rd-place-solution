import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


#--------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("=="*40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [SOE] [LEAD] [CLAIM] [POSITION] [COUNTER_CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [EOE] [LEAD_END] [POSITION_END] [CLAIM_END]')}")

    print("=="*40)
    return tokenizer


#--------------- Processing ---------------------------------------------#

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

TOKEN_MAP_QUERIES = {
    "topic": ["Topic [TOPIC]", "[TOPIC END]"],
    "Lead": ["Lead: Does the lead grab the readers attention and point towards the position? [LEAD]", "[LEAD END]"],
    "Position": ["Position: Does the position state a clear stance closely related to the topic? [POSITION]",
                 "[POSITION END]"],
    "Claim": [
        "Claim: Is the claim closely relevant to the position and backs up the position with specific points or perspectives. Is the claim valid and acceptable? [CLAIM]",
        "[CLAIM END]"],
    "Counterclaim": ["Counterclaim: Is the counterclaim reasonable and relevant to the position? [COUNTER_CLAIM]",
                     "[COUNTER_CLAIM END]"],
    "Rebuttal": ["Rebuttal: Does the rebuttal directly answer and refute the counterclaim? [REBUTTAL]",
                 "[REBUTTAL END]"],
    "Evidence": [
        "Evidence: Is the evidence closely related  to the claim, counterclaim or rebuttal and provides valid examples? [EVIDENCE]",
        "[EVIDENCE END]"],
    "Concluding Statement": [
        "Concluding Statement: Does the concluding statement effectively restates the claim using different wording? [CONCLUDING_STATEMENT]",
        "[CONCLUDING_STATEMENT END]"]
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


NEW_TOKENS = [
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]",
    "[TOPIC]",  # 12808
    "[SOE]",  # 12809
    "[EOE]",  # 12810
    "[LEAD END]",
    "[POSITION END]",
    "[CLAIM END]",
    "[COUNTER_CLAIM END]",
    "[REBUTTAL END]",
    "[EVIDENCE END]",
    "[CONCLUDING_STATEMENT END]",
    "[TOPIC END]",  # 128018
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

    span = [position, position+substring_length]
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


def process_essay(essay_id, essay_text, topic, anno_df, config):
    """insert newly added tokens in the essay text
    """
    tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()
    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        if config['use_queries']:
            s_tok, e_tok = TOKEN_MAP_QUERIES[d_type]
        else:
            s_tok, e_tok = TOKEN_MAP[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4

    #print(f'Prompt: {prompt}')
    #essay_text = "[SOE]" + " [TOPIC] " + topic + " [TOPIC END] " + essay_text + "[EOE]"
    #essay_text = "[SOE]" + " [TOPIC] " + topic + " [TOPIC END] " +  "[PROMPT] " + prompt + " [PROMPT END] " + essay_text + "[EOE]"
    essay_text = "[SOE]" + " [TOPIC] " + topic + " [TOPIC END] "  + essay_text + "[EOE]"
    return essay_text


def process_input_df(anno_df, notes_df, config):
    """pre-process input dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: processed dataframe
    :rtype: pd.DataFrame
    """
    notes_df = deepcopy(notes_df)
    anno_df = deepcopy(anno_df)

    #------------------- Pre-Process Essay Text --------------------------#
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

    print("=="*40)
    print("processing essay text and inserting new tokens at span boundaries")
    notes_df["essay_text"] = notes_df[["essay_id", "essay_text", "topic"]].apply(
        lambda x: process_essay(x[0], x[1], x[2], anno_df, config), axis=1
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


#--------------- Dataset ----------------------------------------------#


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
        print("=="*40)
        print("token maps...")
        print(TOKEN_MAP)
        print("=="*40)

        if self.config["add_new_tokens"]:
            print("adding new tokens...")
            tokens_to_add = []
            for this_tok in NEW_TOKENS:
                 tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
            self.tokenizer.add_tokens(tokens_to_add)
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
        df = process_input_df(df, essay_df, self.config)

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

#--------------- dataset with truncation ---------------------------------------------#


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
                #print(f'Head candidate: {head_candidate}')
            else:
                head_candidate = [pos for pos, this_id in enumerate(example_input_ids) if (
                    (this_id in START_IDS) & (pos <= config["max_length"]-buffer))]
                #print(f'Trunc Head candidate: {head_candidate}')

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
