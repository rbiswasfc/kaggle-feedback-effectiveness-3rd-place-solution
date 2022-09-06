from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding

try:
    from .deb_dataset_prompt import (DISCOURSE_END_TOKENS,
                                     DISCOURSE_START_TOKENS)
except:
    raise ImportError

MASK_PROB = 0.1

print("=="*40)
print(f"If used, the mask aug probability = {MASK_PROB}")
print(f"in dataloader discourse start tokens: {DISCOURSE_START_TOKENS}")
print(f"in dataloader discourse end tokens: {DISCOURSE_END_TOKENS}")
print("=="*40)


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

        weights_factor = None
        if "weights_factor" in features[0].keys():
            weights_factor = [feature["weights_factor"] for feature in features]

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

        if weights_factor is not None:
            batch["weights_factor"] = [ex_weights + [0.0] * (b_max - len(ex_weights)) for ex_weights in weights_factor]

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
        batch = {k: (torch.tensor(v, dtype=torch.float32) if k in ["multitask_labels", "weights_factor"] else torch.tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}
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

        weights_factor = None
        if "weights_factor" in features[0].keys():
            weights_factor = [feature["weights_factor"] for feature in features]

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

        if weights_factor is not None:
            batch["weights_factor"] = [ex_weights + [0.0] * (b_max - len(ex_weights)) for ex_weights in weights_factor]

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
        do_not_mask_tokens = list(do_not_mask_tokens)  # TODO: check me

        pass_gate = [
            [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
        ]
        pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

        # self.tokenizer.mask_token
        # 10% of the time replace token with mask token
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, MASK_PROB)).bool()
        indices_mask = torch.logical_and(indices_mask, pass_gate)
        input_ids[indices_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        batch["input_ids"] = input_ids  # replaced

        # print(batch["weights"])

        batch = {k: (torch.tensor(v, dtype=torch.float32) if k in ["multitask_labels", "weights_factor"] else torch.tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}
        return batch


@dataclass
class CustomDataCollatorWithPaddingUDAAug(DataCollatorWithPadding):
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

        # masked augmentation
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()

        discourse_token_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
        discourse_end_ids = set(self.tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))
        do_not_mask_tokens = discourse_token_ids.union(discourse_end_ids).union(set(self.tokenizer.all_special_ids))
        do_not_mask_tokens = list(do_not_mask_tokens)  # TODO: check me

        pass_gate = [
            [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
        ]
        pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

        # self.tokenizer.mask_token
        # 10% of the time replace token with mask token
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, 0.1)).bool()
        indices_mask = torch.logical_and(indices_mask, pass_gate)
        input_ids[indices_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        batch["augmented_input_ids"] = input_ids  # replaced

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
