import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from tokenizers import AddedToken
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          PreTrainedTokenizerBase, default_data_collator,
                          get_scheduler)
from transformers.models.luke.modeling_luke import (EntityPredictionHead,
                                                    LukeConfig,
                                                    LukeForMaskedLM,
                                                    LukeLMHead)
from transformers.trainer_pt_utils import get_parameter_names

#----------------- Constants ------------------------------------------#
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


def tokenizer_test(tokenizer):
    print("=="*40)
    print(f"tokenizer len: {len(tokenizer)}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [==SOE==] [==SPAN==] [CLAIM]')}")
    print(
        f"tokenizer test: {tokenizer.tokenize('Starts: [==EOE==] [==END==] [POSITION_END]')}")

    print("=="*40)
#--------------- Utils -------------------------------------------------#


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    ap.add_argument('--use_wandb', action='store_true')
    args = ap.parse_args()
    return args


def init_wandb(config):
    project = config["project"]
    run = wandb.init(
        project=project,
        entity='kaggle-clrp',
        config=config,
        name=config['run_name'],
        anonymous="must",
        job_type="Train",
    )
    return run


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#------- Token Insertion ------------------------------------------------#


def insert_tokens(essay_id, essay_text, anno_df):

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


def get_tokenizer(config):
    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])

    # adding new tokens
    print("adding new tokens...")
    tokens_to_add = []
    for this_tok in NEW_TOKENS:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
    tokenizer.add_tokens(tokens_to_add)
    print(f"tokenizer len: {len(tokenizer)}")
    return tokenizer


@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import math

        import torch

        labels = inputs.clone()

        # geometric distribution for spans
        geo_p, lower, upper = 0.15, 1, 12
        len_distrib = [geo_p * (1-geo_p)**(i - lower) for i in range(lower, upper + 1)]
        len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
        lens = list(range(lower, upper + 1))

        masked_indices = []

        for ex_labels in labels:
            mask_num = math.ceil(len(ex_labels) * self.mlm_probability)
            ex_mask = set()
            while len(ex_mask) < mask_num:
                span_len = np.random.choice(lens, p=len_distrib)
                anchor = np.random.choice(len(ex_labels))
                if anchor in ex_mask:
                    continue
                else:
                    left1, right1 = anchor, min(anchor + span_len, len(ex_labels))
                    for i in range(left1, right1):
                        if len(ex_mask) >= mask_num:
                            break
                        ex_mask.add(i)
            ex_mask_bool = [i in ex_mask for i in range(len(ex_labels))]
            masked_indices.append(ex_mask_bool)
        masked_indices = torch.tensor(masked_indices).bool()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        masked_indices = torch.logical_and(masked_indices, ~special_tokens_mask)
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def get_mlm_dataset(corpus, config):

    # ---- stage 1 ---------------------------------#
    stage_one_config = deepcopy(config)
    stage_one_config["model_checkpoint"] = "roberta-base"
    buffer = 6
    stage_one_config["max_length"] = config["max_length"] - buffer
    print(f"max length for stage one tokenization = {stage_one_config['max_length']}")

    tokenizer = get_tokenizer(stage_one_config)
    print("--"*20)
    print("tokenizer test for roberta....")
    tokenizer_test(tokenizer)
    print("--"*20)

    tokenized_corpus = tokenizer(
        corpus,
        padding=False,
        truncation=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
        max_length=stage_one_config["max_length"],
        stride=stage_one_config["stride"],
        return_overflowing_tokens=True,
    )

    print("tokenization with roberta is completed...")

    tokenized_corpus = dict(tokenized_corpus)
    task_dataset = Dataset.from_dict(tokenized_corpus)

    print("first version of task dataset is created...")

    discourse_token_ids = set(tokenizer.convert_tokens_to_ids(DISCOURSE_START_TOKENS))
    discourse_end_ids = set(tokenizer.convert_tokens_to_ids(DISCOURSE_END_TOKENS))

    def process_spans(examples):
        span_head_idxs, span_tail_idxs = [], []
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        for ex_input_ids, ex_offset_mapping in zip(examples["input_ids"], examples["offset_mapping"]):
            example_span_head_idxs = [pos for pos, this_id in enumerate(ex_input_ids) if this_id in discourse_token_ids]
            example_span_tail_idxs = [pos for pos, this_id in enumerate(ex_input_ids) if this_id in discourse_end_ids]

            # handle edge cases
            # 1. A tail occurs before the first head in the sequence due to truncation
            if (len(example_span_tail_idxs) > 0) & (len(example_span_head_idxs) > 0):
                if example_span_tail_idxs[0] < example_span_head_idxs[0]:  # truncation effect
                    example_span_tail_idxs = example_span_tail_idxs[1:]  # shift by one

            # 2. Tail got chopped off due to truncation but the corresponding head is still there
            if len(example_span_tail_idxs) < len(example_span_head_idxs):
                if len(example_span_tail_idxs) + 1 != len(example_span_head_idxs):
                    print("warning...")
                    print(f"tail idxs: {example_span_tail_idxs}")
                    print(f"head idxs: {example_span_head_idxs}")

                if len(ex_input_ids) != stage_one_config["max_length"]:
                    print("Warning...")  # should only happen if input text is truncated
                    print(f"len inputs = {len(ex_input_ids)}, yet hail tail mismatch")
                example_span_tail_idxs.append(stage_one_config["max_length"]-2)  # the token before [SEP] token

            example_span_head_char_start_idxs = [ex_offset_mapping[pos][0] for pos in example_span_head_idxs]
            example_span_tail_char_end_idxs = [ex_offset_mapping[pos][1] for pos in example_span_tail_idxs]

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

    print("detecting the discourse span character indices...")
    task_dataset = task_dataset.map(process_spans, batched=True)
    print("done...")

    def restore_essay_text(examples):
        essay_text = []

        for example_offset_mapping in examples["offset_mapping"]:
            offset_start = example_offset_mapping[0][0]
            offset_end = example_offset_mapping[-1][1]
            essay_text.append(corpus[offset_start: offset_end])
        return {"essay_text": essay_text}

    def update_head_tail_char_idx(examples):
        span_head_char_start_idxs, span_tail_char_end_idxs = [], []

        for example_span_head_char_start_idxs, example_span_tail_char_end_idxs, example_offset_mapping in zip(
                examples["span_head_char_start_idxs"], examples["span_tail_char_end_idxs"], examples["offset_mapping"]):

            offset_start = example_offset_mapping[0][0]
            offset_end = example_offset_mapping[-1][1]

            example_span_head_char_start_idxs = [pos - offset_start for pos in example_span_head_char_start_idxs]
            example_span_tail_char_end_idxs = [pos - offset_start for pos in example_span_tail_char_end_idxs]
            span_head_char_start_idxs.append(example_span_head_char_start_idxs)
            span_tail_char_end_idxs.append(example_span_tail_char_end_idxs)
        return {"span_head_char_start_idxs": span_head_char_start_idxs, "span_tail_char_end_idxs": span_tail_char_end_idxs}

    def get_entity_spans(examples):
        entity_spans = []
        for ex_starts, ex_ends in zip(examples["span_head_char_start_idxs"], examples["span_tail_char_end_idxs"]):
            ex_entity_spans = [tuple([a, b]) for a, b in zip(ex_starts, ex_ends)]
            entity_spans.append(ex_entity_spans)
        return {"entity_spans": entity_spans}

    print("creating fields for luke tokenizer...")
    task_dataset = task_dataset.map(restore_essay_text, batched=True)
    task_dataset = task_dataset.map(update_head_tail_char_idx, batched=True)
    task_dataset = task_dataset.map(get_entity_spans, batched=True)
    print("done..")
    task_dataset = task_dataset.select(range(len(task_dataset)-1))

    #--------------- stage 2 --------------------------------#
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_checkpoint"],
        max_mention_length=config["max_mention_length"]
    )

    # add new tokens
    print("adding new tokens...")
    tokens_to_add = []
    for this_tok in NEW_TOKENS:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
    tokenizer.add_tokens(tokens_to_add)
    print("tokenizer test for luke...")
    tokenizer_test(tokenizer)

    def tokenize_with_entity_spans(example):
        tz = tokenizer(
            example["essay_text"],
            entity_spans=[tuple(t) for t in example["entity_spans"]],
            truncation=True,
            add_special_tokens=True,
            max_length=config["max_length"],
        )
        return tz

    print("tokenization with luke....")
    task_dataset = task_dataset.map(tokenize_with_entity_spans, batched=False)
    print("done...")

    print("computing max length...")
    n = len(task_dataset)
    max_len = 0
    for i in range(n):
        cl = len(task_dataset[i]["input_ids"])
        if cl > max_len:
            max_len = cl
    print(f"max length for luke dataset = {max_len}")

    # split into train and test
    test_pct = config["test_pct"]
    max_train_examples = config["max_train_examples"]
    max_test_examples = config["max_test_examples"]

    test_size = int(len(task_dataset) * test_pct)
    train_size = len(task_dataset) - test_size

    test_size = min(test_size, max_test_examples)
    train_size = min(train_size, max_train_examples)

    print(f"test dataset size = {test_size}")
    print(f"train dataset size = {train_size}")

    downsampled_dataset = task_dataset.train_test_split(
        train_size=train_size, test_size=test_size, seed=config["seed"]
    )

    downsampled_dataset["test"].set_format(
        type=None,
        columns=["input_ids", "attention_mask", "entity_ids", "entity_position_ids", "entity_attention_mask"]
    )

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config["mask_probability"]
    )

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    downsampled_dataset["test"] = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )

    downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
            "masked_entity_ids": "entity_ids",
            "masked_entity_position_ids": "entity_position_ids",
            "masked_entity_attention_mask": "entity_attention_mask",
        }
    )

    return downsampled_dataset, tokenizer


def execute_mlm(config):
    notes_df = pd.read_csv(config["data_path"], low_memory=False)
    anno_df = pd.read_csv(config["annotation_path"], low_memory=False)

    notes_df = notes_df.rename(columns={"id": "essay_id"})
    anno_df = anno_df.rename(columns={"id": "essay_id"})

    notes_df = notes_df.reset_index(drop=True)
    anno_df = anno_df.reset_index(drop=True)

    print(f"# of essays: {len(notes_df)}")

    if config["do_tapt"]:
        print("removing augmented essays for task adaptation...")
        notes_df["original_essay_id"] = notes_df["essay_id"].apply(lambda x: str(x).split("_")[0])
        remove_essay_ids = notes_df[notes_df["original_essay_id"] != notes_df["essay_id"]].essay_id.unique().tolist()
        notes_df = notes_df[~notes_df["essay_id"].isin(remove_essay_ids)].copy()
        notes_df = notes_df.drop(columns=["original_essay_id"])
        notes_df = notes_df.reset_index(drop=True)

        anno_df = anno_df[~anno_df["essay_id"].isin(remove_essay_ids)].copy()
        anno_df = anno_df.reset_index(drop=True)
        print(f"# of essays: {len(notes_df)}")
        print(f"shape of anno df: {anno_df.shape}")

    if args.use_wandb:
        print("initializing wandb run...")
        init_wandb(config)

    if config["debug"]:
        notes_df = notes_df.sample(512)
    print(notes_df.shape)

    os.makedirs(config["output_dir"], exist_ok=True)
    if config["load_dataset_from_cache"]:
        print("attempting to load mlm dataset...")

        mlm_dataset = load_from_disk(os.path.join(config["output_dir"], "mlm_dataset"))
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_checkpoint"],
            max_mention_length=config["max_mention_length"]
        )

        # add new tokens
        print("adding new tokens...")
        tokens_to_add = []
        for this_tok in NEW_TOKENS:
            tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
        tokenizer.add_tokens(tokens_to_add)
    else:
        print("processing input dataframes...")
        notes_df["essay_text"] = notes_df[["essay_id", "essay_text"]].apply(
            lambda x: insert_tokens(x[0], x[1], anno_df), axis=1
        )
        print("done...")
        notes_df = notes_df[["essay_text"]].copy()
        notes_df.sample(128).to_csv("./notes_sample_luke.csv", index=False)
        print(notes_df.head())

        #------------------- Dataset --------------------------------------#
        sampling_fraction = config["sampling_fraction"]
        notes_df = notes_df.sample(int(sampling_fraction*len(notes_df)))
        # create corpus
        all_notes = notes_df["essay_text"].values.tolist()
        corpus = " ".join(all_notes)

        mlm_dataset, tokenizer = get_mlm_dataset(corpus, config)
        # save
        mlm_dataset.save_to_disk(os.path.join(config["output_dir"], "mlm_dataset"))

    print("tokenizer test...")
    tokenizer_test(tokenizer)
    eval_dataset = deepcopy(mlm_dataset['test'])

    # save the dataset

    #------------------ Model -----------------------------------------#
    base_config = LukeConfig.from_pretrained(config["model_checkpoint"])

    if config["load_from_ckpt"]:
        model = LukeForMaskedLM.from_pretrained(config["model_checkpoint"], config=base_config)
    else:
        model = LukeForMaskedLM.from_pretrained(config["model_checkpoint"], config=base_config)  # , config=base_config)
        model.luke.resize_token_embeddings(len(tokenizer))
        model.lm_head = LukeLMHead(base_config)  # TODO: check impact
        model.entity_predictions = EntityPredictionHead(base_config)

    if config["gradient_checkpointing"]:
        print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # optimizer
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    if config["use_bnb_optim"]:
        print("using bnb optimizer....")
        optimizer_kwargs = {
            "betas": (config["beta1"], config["beta2"]),
            "eps": config['eps'],
        }
        optimizer_kwargs["lr"] = config["lr"]
        adam_bnb_optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(config['beta1'], config['beta2']),
            eps=config['eps'],
            lr=config['lr'],
        )
        optimizer = adam_bnb_optim

    else:
        print("using AdamW optimizer....")
        optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"])

    #-------------- dataloaders ------------------#
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config["mask_probability"]
    )

    batch_size = config["batch_size"]
    mlm_dataset["train"].set_format(
        type=None,
        columns=["input_ids", "attention_mask", "entity_ids", "entity_position_ids", "entity_attention_mask"]
    )

    train_dataloader = DataLoader(
        mlm_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        mlm_dataset["test"],
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    print(len(mlm_dataset["train"]), len(mlm_dataset["test"]))

    # accelerator
    accelerator = Accelerator(fp16=config["fp16"])
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    print_gpu_utilization()

    # training setup
    num_train_epochs = config["num_epochs"]

    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    warmup_frac = config["warmup_pct"]

    num_update_steps_per_epoch = len(train_dataloader)//gradient_accumulation_steps
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_frac*num_training_steps)

    print(f"num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    print(f"num_training_steps = {num_training_steps}")
    print(f"num_warmup_steps = {num_warmup_steps}")

    output_dir = config["trained_model_name"]

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    #------------------------------- Training -------------------------------------#
    wandb_step = 0

    for epoch in range(num_train_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_training_steps))
        loss_meter = AverageMeter()

        model.train()

        for step, batch in enumerate(train_dataloader):
            if(step % 100) == 0:
                print(f"current step: {step}")
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                wandb_step += 1
                loss_meter.update(loss.item())

                progress_bar.set_description(
                    f"STEP: {(step+1)//gradient_accumulation_steps:5}/{num_update_steps_per_epoch:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"TL: {loss_meter.avg:.4f}. "
                )

                if args.use_wandb:
                    wandb.log({"Train Loss": loss_meter.avg}, step=wandb_step)
                    wandb.log({"LR": get_lr(optimizer)}, step=wandb_step)

            # Evaluation
            if step % config["eval_frequency"] == 0:
                model.eval()
                losses = []
                n_correct = 0
                n_total = 0

                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                        tok_preds = torch.max(outputs['logits'], dim=-1)[1]
                        tok_labels = batch['labels']
                        curr = torch.masked_select(tok_preds == batch['labels'], batch['labels'] > -100).sum()
                        tot = torch.masked_select(tok_preds == batch['labels'], batch['labels'] > -100).size(0)
                        n_correct += curr
                        n_total += tot

                    loss = outputs.loss
                    losses.append(accelerator.gather(loss.repeat(batch_size)))

                losses = torch.cat(losses)
                losses = losses[: len(eval_dataset)]

                try:
                    perplexity = math.exp(torch.mean(losses))
                except OverflowError:
                    perplexity = float("inf")

                accuracy = round((n_correct*100/n_total).item(), 2)
                print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
                print(f">>> Epoch {epoch}: Accuracy: {accuracy}")

                if args.use_wandb:
                    wandb.log({"Epoch": epoch}, step=wandb_step)
                    wandb.log({"Perplexity": perplexity}, step=wandb_step)
                    wandb.log({"Accuracy": accuracy}, step=wandb_step)

                # Save and upload
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                torch.cuda.empty_cache()
                model.train()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    execute_mlm(config)
