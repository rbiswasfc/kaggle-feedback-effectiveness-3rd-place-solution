import pdb
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
        span_attention_mask = [[1]*len(feature["span_head_idxs"]) for feature in features]
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

        batch["global_attention_mask"] = [
            ex_global_attention_mask + [0] * (max_len - len(ex_global_attention_mask)) for ex_global_attention_mask in global_attention_mask
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


def build_multitask_labels(label_sequence):
    """
    aux_label_0: 0 if ineffective, 1 if adequate, 1 if effective
    aux_label_1: 0 if ineffective, 0 if adequate, 1 if effective
    aux_label_2: 1 if cur_label > prev_label # is_better
    aux_label_3: 1 if cur_label < prev_label # is_worse
    """
    to_return = []

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

    def _is_better(prev, curr):
        if (curr == -1) | (prev == -1):
            return -1
        if curr > prev:
            return 1
        else:
            return 0

    def _is_worse(prev, curr):
        if (curr == -1) | (prev == -1):
            return -1
        if curr < prev:
            return 1
        else:
            return 0

    prev_label = -1
    for cur_label in label_sequence:
        cur_multitask_label = []
        cur_multitask_label.extend(_get_additional_labels(cur_label))
        # is_better
        cur_multitask_label.append(_is_better(cur_label, prev_label))
        # is_worse
        cur_multitask_label.append(_is_worse(cur_label, prev_label))
        prev_label = cur_label
        to_return.append(cur_multitask_label)
    return to_return


@dataclass
class CustomDataCollatorWithPaddingMultiTask(DataCollatorWithPadding):
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

        batch["global_attention_mask"] = [
            ex_global_attention_mask + [0] * (max_len - len(ex_global_attention_mask)) for ex_global_attention_mask in global_attention_mask
        ]

        if labels is not None:
            batch["labels"] = [ex_labels + [-1] * (b_max - len(ex_labels)) for ex_labels in labels]

        if labels is not None:
            additional_labels = [build_multitask_labels(ex_label_sequence) for ex_label_sequence in batch["labels"]]
            batch["multitask_labels"] = additional_labels
        # pdb.set_trace()

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != "multitask_labels" else torch.tensor(
            v, dtype=torch.float32)) for k, v in batch.items()}
        return batch
