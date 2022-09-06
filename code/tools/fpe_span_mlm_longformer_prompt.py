import argparse
import json
import math
import os
import random
import re
import shutil
import string
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from datasets import Dataset
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from tokenizers import AddedToken
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, DataCollatorForLanguageModeling,
                          DataCollatorWithPadding, PreTrainedTokenizerBase,
                          default_data_collator,
                          get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup, get_scheduler)
from transformers.models.longformer.modeling_longformer import (
    LongformerForMaskedLM, LongformerLMHead)
from transformers.trainer_pt_utils import get_parameter_names

tqdm.pandas()

def init_wandb(config):
    project = config["project"]
    run = wandb.init(
        project=project,
        entity='kaggle-clrp',
        #group=f"exp-{str(config['exp_no']).zfill(3)}",
        config=config,
        name=config['run_name'],
        anonymous="must",
        job_type="Train",
    )
    return run

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


def insert_tokens(essay_id, essay_text, topic, anno_df):
    tok_map = {
        "topic": ["[TOPIC]", "[TOPIC END]"],
        "Lead": ["[LEAD]", "[LEAD END]"],
        "Position": ["[POSITION]", "[POSITION END]"],
        "Claim": ["[CLAIM]", "[CLAIM END]"],
        "Counterclaim": ["[COUNTER_CLAIM]", "[COUNTER_CLAIM END]"],
        "Rebuttal": ["[REBUTTAL]", "[REBUTTAL END]"],
        "Evidence": ["[EVIDENCE]", "[EVIDENCE END]"],
        "Concluding Statement": ["[CONCLUDING_STATEMENT]", "[CONCLUDING_STATEMENT END]"]
    }

    try:
        tmp_df = anno_df[anno_df["id"] == essay_id].copy()
    except Exception as e:
        tmp_df = anno_df[anno_df["essay_id"] == essay_id].copy()

    tmp_df = tmp_df.sort_values(by="discourse_start")
    buffer = 0

    for _, row in tmp_df.iterrows():
        s, e, d_type = int(row.discourse_start) + buffer, int(row.discourse_end) + buffer, row.discourse_type
        s_tok, e_tok = tok_map[d_type]
        essay_text = " ".join([essay_text[:s], s_tok, essay_text[s:e], e_tok, essay_text[e:]])
        buffer += len(s_tok) + len(e_tok) + 4
    # essay_text = re.sub(re.compile(r'(\r\n|\r|\n)'), "[BR]", essay_text)
    essay_text = "[SOE]" + " [TOPIC] " + str(topic) + " [TOPIC END] " + essay_text + "[EOE]"
    return essay_text


def get_tokenizer(config):
    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])

    # adding new tokens
    new_tokens = [
        "[LEAD]",
        "[POSITION]",
        "[CLAIM]",
        "[COUNTER_CLAIM]",
        "[REBUTTAL]",
        "[EVIDENCE]",
        "[CONCLUDING_STATEMENT]",
        "[TOPIC]", #12808
        "[SOE]", #12809
        "[EOE]", #12810
        "[LEAD END]",
        "[POSITION END]",
        "[CLAIM END]",
        "[COUNTER_CLAIM END]",
        "[REBUTTAL END]",
        "[EVIDENCE END]",
        "[CONCLUDING_STATEMENT END]",
        "[TOPIC END]", #128018
    ]
    # adding new tokens
    print("adding new tokens...")
    tokens_to_add = []
    for this_tok in new_tokens:
        tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
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
        geo_p, lower, upper = 0.15, 1, 20
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
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def get_mlm_dataset(notes_df, config, tokenizer):
    print(notes_df.head())
    nbme_dataset = Dataset.from_pandas(notes_df)
    # tokenizer = get_tokenizer(config)

    def tokenize_function(examples):
        result = tokenizer(examples[config["text_col"]])
        return result

    tokenized_datasets = nbme_dataset.map(
        tokenize_function, batched=True, remove_columns=[config["text_col"]]
    )

    try:
        tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
    except:
        pass

    print(tokenized_datasets)

    chunk_size = config["chunk_size"]

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    test_pct = config["test_pct"]
    max_train_examples = config["max_train_examples"]
    max_test_examples = int(config["max_train_examples"]*test_pct)

    test_size = int(len(lm_datasets) * test_pct)
    train_size = len(lm_datasets) - test_size

    test_size = min(test_size, max_test_examples)
    train_size = min(train_size, max_train_examples)

    downsampled_dataset = lm_datasets.train_test_split(
        train_size=train_size, test_size=test_size, seed=config["seed"]
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

    try:
        downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
                "masked_token_type_ids": "token_type_ids",
            }
        )
    except:
        downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )

    return downsampled_dataset


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

def execute_mlm(config):
    # load data
    if 'csv' in config["data_path"]:
        notes_df = pd.read_csv(config["data_path"])
    else:
        notes_df = pd.read_parquet(config["data_path"])
    notes_df = notes_df.rename(columns={"id": "essay_id"})
    notes_df = notes_df.reset_index(drop=True)
    if 'csv' in config["data_path"]:
        topics_df = pd.read_csv('t5_topics.csv')
    else:
        topics_df = pd.read_csv('topics.csv')
    topics_df['topic'] = topics_df['topic'].map(topic_map)
    notes_df = notes_df.merge(topics_df, on='essay_id', how='left')
    notes_df = notes_df.reset_index(drop=True)
    print(notes_df.head())

    anno_df = pd.read_csv(config["annotation_path"])

    if args.use_wandb:
        print("initializing wandb run...")
        init_wandb(config)

    if config["debug"]:
        notes_df = notes_df.sample(512)
    print(notes_df.shape)

    notes_df["essay_text"] = notes_df[["essay_id", "essay_text", "topic"]].progress_apply(
        lambda x: insert_tokens(x[0], x[1], x[2], anno_df), axis=1
    )
    notes_df = notes_df[["essay_text"]].copy()
    notes_df.sample(128).to_csv("./notes_sample.csv", index=False)
    print(notes_df.head())
    tokenizer = get_tokenizer(config)

    # model
    base_config = AutoConfig.from_pretrained(config["model_checkpoint"])
    # base_config.update(
    #     {
    #         # "vocab_size": len(tokenizer),
    #         "max_position_embeddings": 1024,

    #     }
    # )
    model = AutoModelForMaskedLM.from_pretrained(config["model_checkpoint"], config=base_config)
    model.longformer.resize_token_embeddings(len(tokenizer))
    model.lm_head = LongformerLMHead(base_config)

    if config["gradient_checkpointing"]:
        print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # optimizer
    if config["use_bnb_optim"]:
        print("using bnb optimizer....")
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
        optimizer = AdamW(model.parameters(), lr=config["lr"])

    # datasets
    sampling_fraction = config["sampling_fraction"]
    notes_df_sample = notes_df.sample(int(sampling_fraction*len(notes_df)))
    mlm_dataset = get_mlm_dataset(notes_df_sample, config, tokenizer)

    eval_dataset = deepcopy(mlm_dataset['test'])

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config["mask_probability"]
    )

    batch_size = config["batch_size"]

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
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if(step % 100) == 0:
                print(f"current step: {step}")
            # set_trace()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(),
                #     100.0
                # )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

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
                    wandb.log({"Epoch": epoch})
                    wandb.log({"Perplexity": perplexity})
                    wandb.log({"Accuracy": accuracy})

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