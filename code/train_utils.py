import argparse
import json
import os
import random
import string

import numpy as np
import torch
import torch.utils.checkpoint
import wandb
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from sklearn.metrics import f1_score, log_loss


def generate_random_hash(size=6):
    chars = string.ascii_lowercase + string.digits
    h = "".join(random.SystemRandom().choice(chars) for _ in range(size))
    return h


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    ap.add_argument("--fold", type=int, required=False, default=0)
    ap.add_argument("--use_wandb", action="store_true", required=False)

    args = ap.parse_args()
    return args


def read_config(args):
    with open(args.config_path, "r") as f:
        config = json.load(f)
    return config


def init_wandb(config):
    project = config["project"]
    tags = config["tags"]
    run_id = f"{config['run_name']}-fold-{config['fold']}"  # -{generate_random_hash()}"
    run = wandb.init(
        project=project,
        entity='kaggle-clrp',
        # group=f"exp-{str(config['exp_no']).zfill(3)}",
        config=config,
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )
    return run


def print_gpu_utilization(device=0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(device))
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def get_score(y_true, y_pred):
    # y_pred = softmax(y_pred)
    score = log_loss(y_true, y_pred)
    return round(score, 5)
