import gc
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from src.fpe_meta.meta_dataloader import CustomDataCollatorWithPadding
    from src.fpe_meta.meta_dataset import FeedbackDatasetMeta
    from src.fpe_meta.meta_model import (FeedbackMetaModelDev,
                                         FeedbackMetaModelResidual,
                                         FeedbackMetaModelSimple)
    from src.fpe_utils import AverageMeter, get_lr, save_checkpoint
    from train_utils import (get_score, init_wandb, parse_args, read_config,
                             seed_everything)

except Exception:
    raise ImportError

#-------- Main Function -----------------------------------------------------#

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

def run_training():
    #-------- seed ------------#
    print("=="*40)
    seed = random.randint(401, 999)
    print(f"setting seed: {seed}")
    seed_everything(seed)

    args = parse_args()
    config = read_config(args)
    config["seed"] = seed

    fold = args.fold
    config["train_folds"] = [i for i in range(config["n_folds"]) if i != fold]
    config["valid_folds"] = [fold]
    config["fold"] = fold
    print(f"train folds: {config['train_folds']}")
    print(f"valid folds: {config['valid_folds']}")
    print("=="*40)

    os.makedirs(config["model_dir"], exist_ok=True)

    # load train data
    df = pd.read_csv(os.path.join(config["fpe_dataset_dir"], "train.csv"))
    fold_df = pd.read_parquet(config["fold_path"])
    df = pd.merge(df, fold_df, on="essay_id", how="left")
    essay_df = pd.read_parquet(config["train_essay_fpe22_dir"])

    oof_dfs = []
    for csv in config["oof_csvs"]:
        oof_df = pd.read_csv(csv)
        oof_dfs.append(oof_df)

    pred_cols = ["Ineffective", "Adequate", "Effective"]
    for model_idx in range(len(oof_dfs)):
        col_map = dict()
        for col in pred_cols:
            col_map[col] = f"model_{model_idx}_{col}"
        oof_dfs[model_idx] = oof_dfs[model_idx].rename(columns=col_map)

    merged_df = oof_dfs[0]

    for oof_df in oof_dfs[1:]:
        keep_cols = ["discourse_id"] + [col for col in oof_df.columns if col.startswith("model")]
        oof_df = oof_df[keep_cols].copy()
        merged_df = pd.merge(merged_df, oof_df, on="discourse_id", how='inner')
    assert merged_df.shape[0] == oof_dfs[0].shape[0]

    feature_names = [col for col in merged_df.columns if col.startswith("model")]
    feature_map = dict(zip(merged_df["discourse_id"], merged_df[feature_names].values))
    feature_names_cat = [col for col in merged_df.columns if col.startswith("freq") or 'type1' in col]

    df_topics = pd.read_csv(config["topic_path"])
    df_topics["topic_num"] = df_topics['topic'].map(topic_map)
    #df_topics["topic_num"] = df_topics["topic_num"] + 1  # added 1 for padding
    #df_topics["topic_num"] = df_topics["topic_num"].clip(lower=0, upper=16)
    topics_map = dict(zip(df_topics["essay_id"], df_topics["topic_num"]))

    config["num_features"] = len(feature_names)
    config["cat_features"] = feature_names_cat

    if config["debug"]:
        print("DEBUG Mode: sampling 1024 examples from train data")
        df = df.sample(min(1024, len(df)))

    all_ids = df["discourse_id"].unique().tolist()
    discourse2idx = {discourse: pos for pos, discourse in enumerate(all_ids)}
    discourse2effectiveness = dict(zip(df["discourse_id"], df["discourse_effectiveness"]))

    idx2discourse = {v: k for k, v in discourse2idx.items()}
    df["uid"] = df["discourse_id"].map(discourse2idx)

    # create the dataset
    print("creating the datasets and data loaders...")
    train_df = df[df["kfold"].isin(config["train_folds"])].copy()
    valid_df = df[df["kfold"].isin(config["valid_folds"])].copy()

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")

    if config["load_dataset_from_disk"]:
        tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])
        train_dataset = load_from_disk(os.path.join(config["model_dir"], f"train_dataset_fold_{fold}"))
        valid_dataset = load_from_disk(os.path.join(config["model_dir"], f"valid_dataset_fold_{fold}"))

    else:
        dataset_creator = FeedbackDatasetMeta(config)
        train_dataset = dataset_creator.get_dataset(train_df, essay_df, feature_map, topics_map, mode="train")
        valid_dataset = dataset_creator.get_dataset(valid_df, essay_df, feature_map, topics_map, mode="valid")
        # save datasets
        train_dataset.save_to_disk(os.path.join(config["model_dir"], f"train_dataset_fold_{fold}"))
        valid_dataset.save_to_disk(os.path.join(config["model_dir"], f"valid_dataset_fold_{fold}"))
        tokenizer = dataset_creator.tokenizer

    config["len_tokenizer"] = len(tokenizer)
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
                 'span_tail_idxs', 'discourse_type_ids', "meta_features", 'labels', 'uids', "topic_nums"]
    )

    # sort valid dataset for faster evaluation
    valid_dataset = valid_dataset.sort("input_length")

    valid_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
                 'span_tail_idxs', 'discourse_type_ids', "meta_features", 'labels', 'uids', "topic_nums"]
    )

    data_collector_train = data_collector

    train_dl = DataLoader(
        train_dataset,
        batch_size=config["train_bs"],
        shuffle=True,
        collate_fn=data_collector_train,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=config["valid_bs"],
        shuffle=False,
        collate_fn=data_collector,
        pin_memory=True,
    )
    print("data preparation done...")
    print("=="*40)

    # create the model and optimizer
    print("creating the model, optimizer and scheduler...")
    if config["use_dev_model"]:
        print("using dev model...")
        model = FeedbackMetaModelDev(config)
    else:
        print("using residual model...")
        model = FeedbackMetaModelResidual(config)

    # prepare the training
    num_epochs = config["num_epochs"]
    warmup_pct = config["warmup_pct"]

    num_update_steps_per_epoch = len(train_dl)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator()
    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    # wandb
    if args.use_wandb:
        print("initializing wandb run...")
        init_wandb(config)

    # training
    best_loss = 1e6
    tracker = 0
    num_epochs = config["num_epochs"]
    wandb_step = 0

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()

        # Training
        model.train()

        for step, batch in enumerate(train_dl):
            # print(batch)
            logits, loss = model(**batch)
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["grad_clip"],
            )

            # take optimizer and scheduler steps
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_meter.update(loss.item())

            progress_bar.set_description(
                f"Epoch: {epoch + 1:3}. "
                f"STEP: {step + 1:5}/{num_update_steps_per_epoch:5}. "
                f"LR: {get_lr(optimizer):.4f}. "
                f"Loss: {loss_meter.avg:.4f}. "
            )

            progress_bar.update(1)
            wandb_step += 1

            if args.use_wandb:
                wandb.log({"Train Loss": loss_meter.avg}, step=wandb_step)
                wandb.log({"LR": get_lr(optimizer)}, step=wandb_step)

            # Evaluation
            if (step + 1) % config["eval_frequency"] == 0:
                model.eval()
                all_preds = []
                all_uids = []
                all_labels = []

                for _, batch in enumerate(valid_dl):
                    with torch.no_grad():
                        logits, loss = model(**batch)
                        batch_preds = F.softmax(logits, dim=-1)
                        batch_uids = batch["uids"]
                        batch_labels = batch["labels"]
                    all_preds.append(batch_preds)
                    all_uids.append(batch_uids)
                    all_labels.append(batch_labels)

                all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
                all_preds = list(chain(*all_preds))
                flat_preds = list(chain(*all_preds))

                all_uids = [p.to('cpu').detach().numpy().tolist() for p in all_uids]
                all_uids = list(chain(*all_uids))
                flat_uids = list(chain(*all_uids))

                all_labels = [p.to('cpu').detach().numpy().tolist() for p in all_labels]
                all_labels = list(chain(*all_labels))
                flat_labels = list(chain(*all_labels))

                preds_df = pd.DataFrame(flat_preds)
                preds_df.columns = ["Ineffective", "Adequate", "Effective"]
                preds_df["span_uid"] = flat_uids  # SORTED_DISCOURSE_IDS
                preds_df = preds_df[preds_df["span_uid"] >= 0].copy()
                preds_df["discourse_id"] = preds_df["span_uid"].map(idx2discourse)
                preds_df = preds_df[["discourse_id", "Ineffective", "Adequate", "Effective"]].copy()
                preds_df = preds_df.groupby("discourse_id")[
                    ["Ineffective", "Adequate", "Effective"]].agg(np.mean).reset_index()
                preds_df["label"] = preds_df["discourse_id"].map(discourse2effectiveness)
                preds_df["label"] = preds_df["label"].map({"Ineffective": 0, "Adequate": 1, "Effective": 2})

                # compute loss
                ave_loss = get_score(preds_df["label"].values, preds_df[["Ineffective", "Adequate", "Effective"]].values)
                # print(f"valid loss = {ave_loss}")

                # save teacher
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': step + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': ave_loss,
                }

                if args.use_wandb:
                    wandb.log({"LB": ave_loss}, step=wandb_step)

                is_best = False
                if ave_loss < best_loss:
                    best_loss = ave_loss
                    is_best = True
                    tracker = 0
                else:
                    tracker += 1

                if is_best:
                    print(f">>> best valid loss = {best_loss} <<<")
                    save_checkpoint(config, model_state, is_best=is_best)
                    preds_df.to_csv(os.path.join(config["model_dir"], f"oof_df_fold_{fold}.csv"), index=False)
                else:
                    if tracker % 20 == 0:
                        print(f"patience reached {tracker}/{config['patience']}")

                if args.use_wandb:
                    wandb.log({"Best LB": best_loss}, step=wandb_step)

                torch.cuda.empty_cache()
                model.train()

                if tracker >= config["patience"]:
                    print("stopping early")
                    model.eval()
                    break


if __name__ == "__main__":
    run_training()
