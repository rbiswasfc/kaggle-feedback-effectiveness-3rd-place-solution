import gc
import os
import pdb
import random
from collections import OrderedDict
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from src.fpe_dexl.dexl_dataloader import (
        CustomDataCollatorWithPadding, CustomDataCollatorWithPaddingMaskAug)
    from src.fpe_dexl.dexl_dataset import get_fast_dataset
    from src.fpe_dexl.dexl_model import AWP, FeedbackModel
    from src.fpe_utils import (AverageMeter, apply_mixout, get_lr,
                               get_optimizer, get_scheduler, save_checkpoint)
    from train_utils import (get_score, init_wandb, parse_args,
                             print_gpu_utilization, read_config,
                             seed_everything)

except Exception:
    raise ImportError

#-------- Main Function -----------------------------------------------------#


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
    if config["use_augmented_data"]:
        config["train_folds"].append(999)

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

    if config["use_augmented_data"]:
        print("using augmented data...")
        aug_essay_df = pd.read_csv(config["augmented_essay_path"])
        aug_essay_df["original_essay_id"] = aug_essay_df["essay_id"].apply(lambda x: str(x).split("_")[0])

        valid_df_pre = df[df["kfold"].isin(config["valid_folds"])].copy().reset_index(drop=True)
        valid_essay_ids = valid_df_pre["essay_id"].unique().tolist()
        print(f"Essay shape before valid removal: {aug_essay_df.shape}")
        aug_essay_df = aug_essay_df[~aug_essay_df["original_essay_id"].isin(valid_essay_ids)].copy()
        print(f"Essay shape after valid removal: {aug_essay_df.shape}")

        aug_essay_df = aug_essay_df.sample(frac=1.0)
        aug_essay_df = aug_essay_df.drop_duplicates(subset=["original_essay_id"])
        aug_essay_df = aug_essay_df.sample(config["augmented_samples"])
        aug_essay_df = aug_essay_df.reset_index(drop=True)
        selected_essays = aug_essay_df["essay_id"].unique().tolist()
        aug_essay_df = aug_essay_df.drop(columns=["original_essay_id"])
        print(f"using {len(aug_essay_df)} augmented essays")

        aug_df = pd.read_csv(config["augmented_annotations_path"])
        aug_df = aug_df[aug_df["essay_id"].isin(selected_essays)].copy()
        aug_df = aug_df.drop(columns=["discourse_start", "discourse_end"])
        aug_df = aug_df.reset_index(drop=True)
        aug_df["kfold"] = 999
        # config["train_folds"].append(999)

        essay_df = pd.concat([essay_df, aug_essay_df]).reset_index(drop=True)
        df = pd.concat([df, aug_df]).reset_index(drop=True)

    if config["debug"]:
        print("DEBUG Mode: sampling 1024 examples from train data")
        df = df.sample(min(1024, len(df)))

    all_ids = df["discourse_id"].unique().tolist()
    discourse2idx = {discourse: pos for pos, discourse in enumerate(all_ids)}
    discourse2effectiveness = dict(zip(df["discourse_id"], df["discourse_effectiveness"]))

    idx2discourse = {v: k for k, v in discourse2idx.items()}
    df["uid"] = df["discourse_id"].map(discourse2idx)

    #------- dataset ------------------------------------------------------------#
    print("creating the datasets and data loaders...")
    train_df = df[df["kfold"].isin(config["train_folds"])].copy()
    valid_df = df[df["kfold"].isin(config["valid_folds"])].copy()

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")

    if config["use_augmented_data"]:
        print("removing some essays to avoid data leakage...")
        valid_essay_ids = valid_df["essay_id"].unique().tolist()
        train_df["original_essay_id"] = train_df["essay_id"].apply(lambda x: str(x).split("_")[0])
        print(f"train_df shape before removal: {train_df.shape}")
        train_df = train_df[~train_df["original_essay_id"].isin(valid_essay_ids)].copy()
        print(f"train_df shape after removal: {train_df.shape}")
        train_df = train_df.drop(columns=["original_essay_id"])
        train_df = train_df.reset_index(drop=True)

    train_ds_dict = get_fast_dataset(config, train_df, essay_df, mode="train")
    valid_ds_dict = get_fast_dataset(config, valid_df, essay_df, mode="valid")

    tokenizer = train_ds_dict["tokenizer"]
    train_dataset = train_ds_dict["dataset"]
    valid_dataset = valid_ds_dict["dataset"]

    # save datasets
    train_ds_dict["original_dataset"].save_to_disk(os.path.join(config["model_dir"], f"train_dataset_fold_{fold}"))
    valid_ds_dict["original_dataset"].save_to_disk(os.path.join(config["model_dir"], f"valid_dataset_fold_{fold}"))

    config["len_tokenizer"] = len(tokenizer)
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    if config["use_mask_aug"]:
        data_collector_train = CustomDataCollatorWithPaddingMaskAug(tokenizer=tokenizer)
    else:
        data_collector_train = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
                 'span_tail_idxs', 'discourse_type_ids', 'labels', 'uids']
    )

    # sort valid dataset for faster evaluation
    valid_dataset = valid_dataset.sort("input_length")

    valid_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'span_head_idxs',
                 'span_tail_idxs', 'discourse_type_ids', 'labels', 'uids']
    )

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
    model = FeedbackModel(config)

    if config.get("resume_from_checkpoint", False):
        print("loading model from previously trained checkpoint...")
        checkpoint = config["ckpt_path"]
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        print(f"At onset model performance on validation set = {ckpt['loss']}")
        del ckpt
        gc.collect()

    optimizer = get_optimizer(model, config)

    # prepare the training
    num_epochs = config["num_epochs"]
    grad_accumulation_steps = config["grad_accumulation"]
    warmup_pct = config["warmup_pct"]

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)

    AWP_FLAG = False
    SWA_FLAG = False

    if config.get("resume_from_checkpoint", False):
        AWP_FLAG = True

    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"], anneal_epochs=config["swa_anneal_epochs"])
    swa_model = AveragedModel(model)

    # AWP
    if config["use_awp"]:
        awp = AWP(model, optimizer, adv_lr=config["adv_lr"], adv_eps=config["adv_eps"])
        assert config["grad_accumulation"] == 1, "Grad accumulation not supported with AWP"

    # accelerator
    if config["use_fp16"]:
        print("using mixed precision training")
        accelerator = Accelerator(fp16=True)  # (fp16=True)
    else:
        accelerator = Accelerator()  # (fp16=True)

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print("=="*40)

    # wandb
    if args.use_wandb:
        print("initializing wandb run...")
        init_wandb(config)
        # Save all files that currently exist containing the substring "ckpt"
        wandb.save(f"{config['code_dir']}/*")
        wandb.save(f"{config['code_dir']}/src/*")
        wandb.save(f"{config['code_dir']}/src/fpe_fast/*")

        wandb.save(f"{config['code_dir']}/configs/*")
        wandb.watch(model, log_freq=10)

    # training
    best_loss = 1e6
    save_trigger = config["save_trigger"]
    tracker = 0
    wandb_step = 0
    org_eval_freq = config["eval_frequency"]

    for epoch in range(num_epochs):
        if epoch >= 5:
            print("!!! stopping training at onset of epoch 5 !!!")
            break
        if (config["use_awp"]) & (epoch >= config["awp_trigger_epoch"]):
            print("AWP is triggered...")
            AWP_FLAG = True

        if epoch >= config["swa_trigger_epoch"]:
            print("SWA is triggered...")
            SWA_FLAG = True

        if epoch < config["full_eval_start_epoch"]:
            config["eval_frequency"] = org_eval_freq * 4
        else:
            config["eval_frequency"] = org_eval_freq

        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        multitask_loss_meter = AverageMeter()

        # Training
        model.train()

        for step, batch in enumerate(train_dl):
            logits, loss, loss_dict = model(**batch)
            # pdb.set_trace()

            if args.use_wandb:
                wandb.log({"Loss": loss.item()}, step=wandb_step)

            accelerator.backward(loss)

            # # Track gradient after clipping
            # max_g, min_g = 0, 0
            # for param in model.parameters():
            #     cur_max = torch.amax(param.grad)
            #     cur_min = torch.amin(param.grad)
            #     if cur_max > max_g:
            #         max_g = cur_max
            #     if cur_min < min_g:
            #         min_g = cur_min

            # if args.use_wandb:
            #     wandb.log({"MaxGrad": max_g.item()}, step=wandb_step)
            #     wandb.log({"MinGrad": min_g.item()}, step=wandb_step)

            if AWP_FLAG:
                awp.attack_backward(batch, accelerator)

            if (step + 1) % grad_accumulation_steps == 0:
                if config["use_fp16"]:
                    pass  # not doing grad clipping in mixed precision mode
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["grad_clip"],
                    )

                # take optimizer and scheduler steps
                optimizer.step()

                if not SWA_FLAG:
                    scheduler.step()
                else:
                    if (step + 1) % config["swa_step_frequency"] == 0:
                        print("taking SWA scheduler step...")
                        swa_scheduler.step()

                optimizer.zero_grad()

                loss_meter.update(loss.item())
                try:
                    ce_loss_meter.update(loss_dict["ce_loss"].item())
                    multitask_loss_meter.update(loss_dict["multitask_loss"].item())
                except Exception as e:
                    print(e)
                    ce_loss_meter.update(0.)
                    multitask_loss_meter.update(0.)

                progress_bar.set_description(
                    f"STEP: {(step+1)//grad_accumulation_steps:5}/{num_update_steps_per_epoch:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"TL: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                wandb_step += 1

                if args.use_wandb:
                    wandb.log({"Train Loss": loss_meter.avg}, step=wandb_step)
                    wandb.log({"CE Loss": ce_loss_meter.avg}, step=wandb_step)
                    wandb.log({"MTL Loss": multitask_loss_meter.avg}, step=wandb_step)
                    wandb.log({"LR": get_lr(optimizer)}, step=wandb_step)

            # Evaluation
            if (step + 1) % config["eval_frequency"] == 0:  # ) & (epoch >= config["eval_epoch_start"]):
                print("\n")
                print("GPU Utilization before evaluation...")
                print("\n")
                print_gpu_utilization()

                model.eval()
                all_preds = []
                all_uids = []
                all_labels = []

                for _, batch in enumerate(valid_dl):
                    with torch.no_grad():
                        logits, loss, _ = model(**batch)
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
                print(f"valid loss = {ave_loss}")

                if args.use_wandb:
                    wandb.log({"LB": ave_loss}, step=wandb_step)

                # save teacher
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': step + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': ave_loss,
                }

                is_best = False
                if ave_loss < best_loss:
                    best_loss = ave_loss
                    is_best = True
                    tracker = 0
                else:
                    tracker += 1

                if is_best:
                    if best_loss < save_trigger:
                        save_checkpoint(config, model_state, is_best=is_best)
                    preds_df.to_csv(os.path.join(config["model_dir"], f"oof_df_fold_{fold}.csv"), index=False)
                else:
                    print(f"patience reached {tracker}/{config['patience']}")

                if args.use_wandb:
                    wandb.log({"Best LB": best_loss}, step=wandb_step)

                if best_loss < config["swa_trigger"]:
                    print("SWA is triggered...")
                    SWA_FLAG = True

                if (config["use_awp"]) & (best_loss <= config["awp_trigger"]):
                    print("AWP is triggered...")
                    AWP_FLAG = True

                if SWA_FLAG:
                    # print("making SWA scheduler step...")
                    # swa_scheduler.step()
                    # update average model
                    if ave_loss < config["swa_save_trigger"]:
                        model.to('cpu')
                        swa_model.update_parameters(model)
                        swa_model_name = f"swa_model_fold_{fold}"
                        swa_filename = f'{config["model_dir"]}/{swa_model_name}_model.pth.tar'
                        swa_state = {
                            'step': step + 1,
                            'state_dict': swa_model.state_dict(),
                        }
                        print("saving swa state...")
                        # if is_best:
                        # torch.save(swa_state, swa_filename, _use_new_zipfile_serialization=False)
                        model = accelerator.prepare(model)

                torch.cuda.empty_cache()
                model.train()
                print("GPU Utilization after evaluation...")
                print_gpu_utilization()
                print("=="*40)

                if tracker >= config["patience"]:
                    print("stopping early")
                    model.eval()
                    break


if __name__ == "__main__":
    run_training()
