from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding
import numpy as np


# from dexl_dataset import DISCOURSE_END_TOKENS, DISCOURSE_START_TOKENS
DISCOURSE_START_TOKENS = [
    "[SOE]",
    "[TOPIC]",
    "[LEAD]",
    "[POSITION]",
    "[CLAIM]",
    "[COUNTER_CLAIM]",
    "[REBUTTAL]",
    "[EVIDENCE]",
    "[CONCLUDING_STATEMENT]"
]

DISCOURSE_END_TOKENS = [
    "[EOE]",
    "[TOPIC END]",
    "[LEAD END]",
    "[POSITION END]",
    "[CLAIM END]",
    "[COUNTER_CLAIM END]",
    "[REBUTTAL END]",
    "[EVIDENCE END]",
    "[CONCLUDING_STATEMENT END]",
]

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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]

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
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
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


@dataclass
class CustomDataCollatorWithPaddingPseudo(DataCollatorWithPadding):
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]
        confidence_scores = [feature["confidence_scores"] for feature in features]
        sd_scores = [feature["sd_scores"] for feature in features]

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
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]

        batch["confidence_scores"] = [
            ex_confidence_scores + [0.] * (b_max - len(ex_confidence_scores)) for ex_confidence_scores in confidence_scores
        ]

        batch["sd_scores"] = [
            ex_sd_scores + [0.] * (b_max - len(ex_sd_scores)) for ex_sd_scores in sd_scores
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
        batch = {k: (torch.tensor(v, dtype=torch.int64) if k not in ["multitask_labels", "confidence_scores", "sd_scores"] else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


@dataclass
class CustomDataCollatorWithPaddingRandAug(DataCollatorWithPadding):
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]

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
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
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

        # modify input ids
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # 25% of the time replace my random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.05)).bool()
        indices_random = torch.logical_and(indices_random, ~special_tokens_mask)
        random_tokens = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]
        batch["input_ids"] = input_ids  # replaced

        # batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


@dataclass
class CustomDataCollatorWithPaddingUDA(DataCollatorWithPadding):
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]

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
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
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


@dataclass
class CustomDataCollatorWithPaddingUDARandAug(DataCollatorWithPadding):
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]

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
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
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

        # modify input ids
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # 25% of the time replace my random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.3)).bool()
        indices_random = torch.logical_and(indices_random, ~special_tokens_mask)
        random_tokens = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]
        batch["augmented_input_ids"] = input_ids

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


@dataclass
class CustomDataCollatorWithPaddingMaskAug(DataCollatorWithPadding):
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]

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
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
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

        # masked augmentation
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()

        discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        do_not_mask_tokens = discourse_token_ids.union(discourse_end_ids).union(set(self.tokenizer.all_special_ids))
        do_not_mask_tokens = list(do_not_mask_tokens)

        pass_gate = [
            [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
        ]
        pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

        # self.tokenizer.mask_token
        # 10% of the time replace token with mask token
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, 0.10)).bool()
        indices_mask = torch.logical_and(indices_mask, pass_gate)
        input_ids[indices_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        batch["input_ids"] = input_ids  # replaced
        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch


@dataclass
class CustomDataCollatorWithPaddingMaskAugCutmix(DataCollatorWithPadding):
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]

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

        # Cutmix aug: https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313235
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()
        mask = torch.tensor(batch["attention_mask"])
        print(batch["labels"])
        mask = torch.tensor(batch["attention_mask"])

        print(f'Batch {batch}')
        print(f'shape {input_ids.shape[0]} {input_ids.shape[1]}')

        if np.random.uniform() < 0.5:
            cut = 0.25
            input_ids = input_ids.cpu().numpy()
            perm = torch.randperm(input_ids.shape[0]).cuda()
            rand_len = int(input_ids.shape[1] * cut)
            start = np.random.randint(input_ids.shape[1] - int(input_ids.shape[1] * cut))
            print(f'start {start} perm: {perm} rand_len: {rand_len}')
            input_ids = torch.tensor(input_ids)
            input_ids[:, start:start + rand_len] = input_ids[perm, start:start + rand_len]
            mask[:, start:start + rand_len] = mask[perm, start:start + rand_len]
            #perm = perm.cpu().numpy()
            #print(f'start {start} perm: {perm} rand_len: {rand_len}')
            labels[:, start:start + rand_len] = labels[perm, start:start + rand_len]
            batch["input_ids"] = input_ids.cpu().numpy()  # replaced
            batch["labels"] = labels  # replaced
            batch["attention_mask"] = mask.cpu().numpy()  # replaced

        default_head_idx = max(max_len - 10, 1)  # for padding
        default_tail_idx = max(max_len - 4, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in span_head_idxs
        ]

        batch["uids"] = [ex_uids + [-1] * (b_max - len(ex_uids)) for ex_uids in uids]
        batch["discourse_type_ids"] = [ex_discourse_type_ids + [0] *
                                       (b_max - len(ex_discourse_type_ids)) for ex_discourse_type_ids in discourse_type_ids]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in span_tail_idxs
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

        # masked augmentation
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()

        discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        do_not_mask_tokens = discourse_token_ids.union(discourse_end_ids).union(set(self.tokenizer.all_special_ids))
        do_not_mask_tokens = list(do_not_mask_tokens)

        pass_gate = [
            [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
        ]
        pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

        # self.tokenizer.mask_token
        # 10% of the time replace token with mask token
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, 0.02)).bool()
        indices_mask = torch.logical_and(indices_mask, pass_gate)
        input_ids[indices_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        batch["input_ids"] = input_ids  # replaced
        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}

        return batch