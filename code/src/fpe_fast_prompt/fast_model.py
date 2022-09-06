import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Attention, StableDropout)

#-------- Focal Loss --------------------------------------------------------#


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class FocalLossMPL(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100, reduction='mean'):
        super(FocalLossMPL, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        # loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        loss = F.nll_loss(logpt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)

        return loss
#-------- AWP ------------------------------------------------------------#


class AWP:
    """Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, accelerator):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        _, adv_loss, _ = self.model(**batch)
        self.optimizer.zero_grad()
        accelerator.backward(adv_loss)
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AwpUDA:
    """Implements weighted adverserial perturbation for uda model
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, train_b, original_unlabelled_b, augmented_unlabelled_b, accelerator):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        _, adv_loss, _ = self.model(train_b, original_unlabelled_b, augmented_unlabelled_b)
        self.optimizer.zero_grad()
        accelerator.backward(adv_loss)
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

#-------- Re-initialization ------------------------------------------------------#


def reinit_deberta(base_model, num_reinit_layers):
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


#-------- Model ------------------------------------------------------------------#
class FeedbackModel(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModel, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        if 'v2-xl' in self.config["base_model_path"]:
            base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        else:
            base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        if 'v2-xl' in self.config["base_model_path"]:
            self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)
        else:
            self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])
        if self.config['MSD']:
            self.dropout1 = nn.Dropout(self.config["dropout"]+0.1)
            self.dropout2 = nn.Dropout(self.config["dropout"]+0.2)
            self.dropout3 = nn.Dropout(self.config["dropout"]+0.3)
            self.dropout4 = nn.Dropout(self.config["dropout"]+0.4)

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # self.transform = nn.Sequential(
        #     nn.Linear(feature_size, feature_size),
        #     nn.Tanh(),
        #     LayerNorm(feature_size, self.base_model.config.layer_norm_eps),
        # )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=self.config["focal_gamma"], ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        if self.config['MSD']:
            feature_vector1 = self.dropout(feature_vector)
            feature_vector2 = self.dropout1(feature_vector)
            feature_vector3 = self.dropout2(feature_vector)
            feature_vector4 = self.dropout3(feature_vector)
            feature_vector5 = self.dropout4(feature_vector)

            logits1 = self.classifier(feature_vector1)
            logits2 = self.classifier(feature_vector2)
            logits3 = self.classifier(feature_vector3)
            logits4 = self.classifier(feature_vector4)
            logits5 = self.classifier(feature_vector5)

            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))

            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5) / 5

            focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))

            focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5) / 5
        else:
            feature_vector = self.dropout(feature_vector)
            logits = self.classifier(feature_vector)
            ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

            
        # feature_vector = self.fpe_lstm_layer(feature_vector)[0]  # LSTM layer outputs
        # feature_vector = self.transform(feature_vector)

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None

        # pdb.set_trace()

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits = logits[:, :, :self.num_original_labels]
            multitask_logits = logits[:, :, self.num_original_labels:]
            multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
            multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            loss_dict["multitask_loss"] = multitask_loss
            # assign ce_logits to logits to continue usual workflow
            logits = ce_logits

        loss_dict["ce_loss"] = ce_loss
        loss_dict["focal_loss"] = focal_loss

        if self.config["use_multitask"]:
            loss = ce_loss + focal_loss + multitask_loss
            # loss = ce_loss + multitask_loss
        else:
            loss = ce_loss + focal_loss
            # loss = ce_loss

        return logits, loss, loss_dict


# -------- Model ------------------------------------------------------------------#
class FeedbackModelSlidingWindow(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackModelSlidingWindow, self).__init__()
        self.config = config
        self.window_size = 512
        self.edge_len = 64
        self.inner_len = self.window_size - self.edge_len * 2
        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        if 'v2-xl' in self.config["base_model_path"]:
            self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)
        else:
            self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])
        if self.config['MSD']:
            self.dropout1 = nn.Dropout(self.config["dropout"] + 0.1)
            self.dropout2 = nn.Dropout(self.config["dropout"] + 0.2)
            self.dropout3 = nn.Dropout(self.config["dropout"] + 0.3)
            self.dropout4 = nn.Dropout(self.config["dropout"] + 0.4)

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit']} layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("==" * 40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # self.transform = nn.Sequential(
        #     nn.Linear(feature_size, feature_size),
        #     nn.Tanh(),
        #     LayerNorm(feature_size, self.base_model.config.layer_norm_eps),
        # )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=self.config["focal_gamma"], ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
            self,
            input_ids,
            attention_mask,
            span_head_idxs,
            span_tail_idxs,
            span_attention_mask,
            labels=None,
            multitask_labels=None,
            **kwargs
    ):
        B, L = input_ids.shape

        if L <= self.window_size:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        else:
            segments = (L - self.window_size) // self.inner_len
            if (L - self.window_size) % self.inner_len > self.edge_len:
                segments += 1
            elif segments == 0:
                segments += 1
            x = self.base_model(input_ids=input_ids[:, :self.window_size],
                              attention_mask=attention_mask[:, :self.window_size], return_dict=False)[0]
            for i in range(1, segments + 1):
                start = self.window_size - self.edge_len + (i - 1) * self.inner_len
                end = self.window_size - self.edge_len + (i - 1) * self.inner_len + self.window_size
                end = min(end, L)
                x_next = input_ids[:, start:end]
                mask_next = attention_mask[:, start:end]
                x_next = self.base_model(input_ids=x_next, attention_mask=mask_next, return_dict=False)[0]
                if i == segments:
                    x_next = x_next[:, self.edge_len:]
                else:
                    x_next = x_next[:, self.edge_len:self.edge_len + self.inner_len]
                x = torch.cat([x, x_next], 1)
            outputs = x

        bs = input_ids.shape[0]  # batch size

        encoder_layer = outputs
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head + 1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        if self.config['MSD']:
            feature_vector1 = self.dropout(feature_vector)
            feature_vector2 = self.dropout1(feature_vector)
            feature_vector3 = self.dropout2(feature_vector)
            feature_vector4 = self.dropout3(feature_vector)
            feature_vector5 = self.dropout4(feature_vector)

            logits1 = self.classifier(feature_vector1)
            logits2 = self.classifier(feature_vector2)
            logits3 = self.classifier(feature_vector3)
            logits4 = self.classifier(feature_vector4)
            logits5 = self.classifier(feature_vector5)

            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))

            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5) / 5

            focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))

            focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5) / 5
        else:
            feature_vector = self.dropout(feature_vector)
            logits = self.classifier(feature_vector)
            ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

        # feature_vector = self.fpe_lstm_layer(feature_vector)[0]  # LSTM layer outputs
        # feature_vector = self.transform(feature_vector)

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None

        # pdb.set_trace()

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits = logits[:, :, :self.num_original_labels]
            multitask_logits = logits[:, :, self.num_original_labels:]
            multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
            multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1),
                                                 multitask_labels.view(-1, 1) > -1).mean()

            loss_dict["multitask_loss"] = multitask_loss
            # assign ce_logits to logits to continue usual workflow
            logits = ce_logits

        loss_dict["ce_loss"] = ce_loss
        loss_dict["focal_loss"] = focal_loss

        if self.config["use_multitask"]:
            loss = ce_loss + focal_loss + multitask_loss
            # loss = ce_loss + multitask_loss
        else:
            loss = ce_loss + focal_loss
            # loss = ce_loss

        return logits, loss, loss_dict


class FeedbackModelPseudo(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelPseudo, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
            print("=="*40)

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])
        if self.config['MSD']:
            self.dropout1 = nn.Dropout(self.config["dropout"]+0.1)
            self.dropout2 = nn.Dropout(self.config["dropout"]+0.2)
            self.dropout3 = nn.Dropout(self.config["dropout"]+0.3)
            self.dropout4 = nn.Dropout(self.config["dropout"]+0.4)

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # self.transform = nn.Sequential(
        #     nn.Linear(feature_size, feature_size),
        #     nn.Tanh(),
        #     LayerNorm(feature_size, self.base_model.config.layer_norm_eps),
        # )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=1.0, ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        # feature_vector = mean_feature_vector
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None
        loss_dict["focal_loss"] = None
        loss = None

        # pdb.set_trace()

        if labels is not None:
            if self.config["use_multitask"]:
                # split logits into ce_logits and bce_logits
                ce_logits = logits[:, :, :self.num_original_labels]
                multitask_logits = logits[:, :, self.num_original_labels:]
                multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
                multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                loss_dict["multitask_loss"] = multitask_loss
                # assign ce_logits to logits to continue usual workflow
                logits = ce_logits

            ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

            loss_dict["ce_loss"] = ce_loss
            loss_dict["focal_loss"] = focal_loss

            if self.config["use_multitask"]:
                loss = ce_loss + focal_loss + multitask_loss
                # loss = ce_loss + multitask_loss
            else:
                loss = ce_loss + focal_loss
                # loss = ce_loss

        return logits, loss, loss_dict


class FeedbackModelPseudoNew(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelPseudo, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
            print("=="*40)

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])
        if self.config['MSD']:
            self.dropout1 = nn.Dropout(self.config["dropout"]+0.1)
            self.dropout2 = nn.Dropout(self.config["dropout"]+0.2)
            self.dropout3 = nn.Dropout(self.config["dropout"]+0.3)
            self.dropout4 = nn.Dropout(self.config["dropout"]+0.4)

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # self.transform = nn.Sequential(
        #     nn.Linear(feature_size, feature_size),
        #     nn.Tanh(),
        #     LayerNorm(feature_size, self.base_model.config.layer_norm_eps),
        # )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=1.0, ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        # feature_vector = mean_feature_vector
        if self.config['MSD']:
            feature_vector1 = self.dropout(feature_vector)
            feature_vector2 = self.dropout1(feature_vector)
            feature_vector3 = self.dropout2(feature_vector)
            feature_vector4 = self.dropout3(feature_vector)
            feature_vector5 = self.dropout4(feature_vector)

            logits1 = self.classifier(feature_vector1)
            logits2 = self.classifier(feature_vector2)
            logits3 = self.classifier(feature_vector3)
            logits4 = self.classifier(feature_vector4)
            logits5 = self.classifier(feature_vector5)

            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))

            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5) / 5

            focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))

            focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5) / 5
        else:
            feature_vector = self.dropout(feature_vector)
            logits = self.classifier(feature_vector)
            ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None
        loss_dict["focal_loss"] = None
        loss = None

        # pdb.set_trace()

        if labels is not None:
            if self.config["use_multitask"]:
                # split logits into ce_logits and bce_logits
                ce_logits = logits[:, :, :self.num_original_labels]
                multitask_logits = logits[:, :, self.num_original_labels:]
                multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
                multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                loss_dict["multitask_loss"] = multitask_loss
                # assign ce_logits to logits to continue usual workflow
                logits = ce_logits

            loss_dict["ce_loss"] = ce_loss
            loss_dict["focal_loss"] = focal_loss

            if self.config["use_multitask"]:
                loss = ce_loss + focal_loss + multitask_loss
                # loss = ce_loss + multitask_loss
            else:
                loss = ce_loss + focal_loss
                # loss = ce_loss

        return logits, loss, loss_dict

# class FeedbackModel(nn.Module):
#     """
#     The feedback prize effectiveness model for fast approach
#     """

#     def __init__(self, config):
#         print("=="*40)
#         print("initializing the feedback model...")

#         super(FeedbackModel, self).__init__()
#         self.config = config

#         # base transformer
#         base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
#         base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
#         self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

#         # resize model embeddings
#         print("resizing model embeddings...")
#         print(f"tokenizer length = {config['len_tokenizer']}")
#         self.base_model.resize_token_embeddings(config["len_tokenizer"])

#         # enable gradient checkpointing
#         self.base_model.gradient_checkpointing_enable()

#         # dropouts
#         self.dropout = StableDropout(self.config["dropout"])

#         if config["num_layers_reinit"] > 0:
#             print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
#             reinit_deberta(self.base_model, self.config["num_layers_reinit"])
#             print("=="*40)

#         self.num_labels = self.num_original_labels = self.config["num_labels"]

#         if self.config["use_multitask"]:
#             print("using multi-task approach...")
#             self.num_labels += self.config["num_additional_labels"]

#         # multi-head attention
#         attention_config = deepcopy(self.base_model.config)
#         attention_config.update({"relative_attention": False})
#         self.fpe_span_attention = DebertaV2Attention(attention_config)

#         # classification
#         hidden_size = self.base_model.config.hidden_size
#         feature_size = hidden_size
#         self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

#         # # RNN Head
#         self.fpe_rnn = nn.GRU(
#             input_size=feature_size,
#             hidden_size=hidden_size//2,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True,
#         )

#         # # discourse type embeddings
#         # self.num_dt_embeddings = 8
#         # self.dt_embedding_dim = 16
#         # self.dt_embeddings = nn.Embedding(self.num_dt_embeddings, self.dt_embedding_dim, padding_idx=0)
#         # self.dt_embeddings.weight.data.normal_(mean=0.0, std=self.base_model.config.initializer_range)

#         # self.fpe_dt_rnn = nn.GRU(
#         #     input_size=self.dt_embedding_dim,
#         #     hidden_size=self.dt_embedding_dim//2,
#         #     num_layers=1,
#         #     batch_first=True,
#         #     bidirectional=True,
#         # )
#         # self.dt_layer_norm = LayerNorm(self.dt_embedding_dim, self.base_model.config.layer_norm_eps)

#         # feature_size += self.dt_embedding_dim
#         self.classifier = nn.Linear(feature_size, self.num_labels)

#         # Loss function
#         self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
#         self.focal_loss_fn = FocalLoss(gamma=3.0, ignore_index=-1)
#         self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         span_head_idxs,
#         span_tail_idxs,
#         discourse_type_ids,
#         span_attention_mask,
#         labels=None,
#         multitask_labels=None,
#         **kwargs
#     ):

#         bs = input_ids.shape[0]  # batch size

#         outputs = self.base_model(input_ids, attention_mask=attention_mask)
#         encoder_layer = outputs[0]
#         self.fpe_rnn.flatten_parameters()
#         encoder_layer = self.fpe_rnn(encoder_layer)[0]  # LSTM layer outputs

#         mean_feature_vector = []

#         for i in range(bs):  # TODO: vectorize
#             span_vec_i = []

#             for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
#                 # span feature
#                 tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
#                 span_vec_i.append(tmp)
#             span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
#             mean_feature_vector.append(span_vec_i)

#         mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
#         mean_feature_vector = self.layer_norm(mean_feature_vector)

#         # attend to other features
#         extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
#         span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
#         span_attention_mask = span_attention_mask.byte()
#         feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

#         # # discourse type sequence feature
#         # dt_encodings = self.dt_embeddings(discourse_type_ids)  # (b, nd, h_dt)
#         # dt_encodings = self.dt_layer_norm(dt_encodings)
#         # self.fpe_dt_rnn.flatten_parameters()
#         # dt_encodings = self.fpe_dt_rnn(dt_encodings)[0]  # LSTM layer outputs

#         # feature_vector = torch.cat([feature_vector, dt_encodings], dim=-1)

#         feature_vector = self.dropout(feature_vector)  # dropout

#         # compute logits and loss
#         loss_dict = dict()
#         loss_dict["multitask_loss"] = None
#         loss_dict["ce_loss"] = None

#         logits = self.classifier(feature_vector)
#         # pdb.set_trace()

#         if self.config["use_multitask"]:
#             # split logits into ce_logits and bce_logits
#             ce_logits = logits[:, :, :self.num_original_labels]
#             multitask_logits = logits[:, :, self.num_original_labels:]
#             multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
#             multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

#             loss_dict["multitask_loss"] = multitask_loss
#             # assign ce_logits to logits to continue usual workflow
#             logits = ce_logits

#         ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
#         focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

#         loss_dict["ce_loss"] = ce_loss

#         if self.config["use_multitask"]:
#             loss = ce_loss + focal_loss + multitask_loss
#             # loss = ce_loss + multitask_loss
#         else:
#             loss = ce_loss + focal_loss
#             # loss = ce_loss

#         return logits, loss, loss_dict


#---------------- Feedback Model for UDA --------------------------------#


class FeedbackModelUDA(nn.Module):
    """The feedback prize effectiveness baseline model with UDA
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelUDA, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        # self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        # self.focal_loss_fn = FocalLoss(gamma=3.0, ignore_index=-1)
        # self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.focal_loss_fn = FocalLossMPL(gamma=3.0, ignore_index=-1, reduction='none')
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def get_logits(self, batch):
        """obtain classification logits for the current batch
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        span_head_idxs = batch["span_head_idxs"]
        span_tail_idxs = batch["span_tail_idxs"]
        span_attention_mask = batch["span_attention_mask"]

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        return logits

    def get_tsa_thresh(self, batch_idx):
        training_progress = float(batch_idx) / float(self.config["num_tsa_steps"])
        if self.config["tsa_schedule"] == 'linear_schedule':
            threshold = training_progress
        elif self.config["tsa_schedule"] == 'exp_schedule':
            scale = 5
            threshold = torch.exp((training_progress - 1) * scale)
        elif self.config["tsa_schedule"] == 'log_schedule':
            scale = 5
            threshold = 1 - torch.exp((-training_progress) * scale)
        output = min(threshold * (self.config["tsa_start"] - self.config["tsa_end"]) + self.config["tsa_start"], 1)
        return output

    def train_step(self, labelled_b, original_unlabelled_b, augmented_unlabelled_b, batch_idx):
        # initialize losses
        loss_dict = dict()

        # get labelled logits
        labelled_logits = self.get_logits(labelled_b)

        # UDA Loss
        unlabelled_discourse_mask = original_unlabelled_b["span_attention_mask"].float()  # (b, num_discourse)
        with torch.no_grad():
            uw_logits = self.get_logits(original_unlabelled_b)[:, :, :self.num_original_labels]  # (b, num_discourse, 3)
            uw_probs = F.softmax(uw_logits, dim=-1)  # (b*num_discourse, 3) # KLdiv target
            confidence_mask = torch.max(uw_probs, dim=-1)[0].ge(self.config["confidence_threshold"]).float()

        us_logits = self.get_logits(augmented_unlabelled_b)[:, :, :self.num_original_labels]
        us_probs = F.log_softmax(us_logits/self.config["temperature"], dim=-1)  # (b, num_discourse, 3)

        uda_criterion = nn.KLDivLoss(reduction='none')
        uda_loss = torch.sum(torch.sum(uda_criterion(us_probs, uw_probs), dim=-1)
                             * confidence_mask * unlabelled_discourse_mask)
        uda_loss = uda_loss / torch.sum(confidence_mask*unlabelled_discourse_mask).clamp(min=1.0)
        # weight_uda = self.config["lambda_uda"] * min(1., (batch_idx + 1) / self.config["uda_steps"])
        # uda_loss = uda_loss*weight_uda
        loss_dict["uda_loss"] = uda_loss

        # Supervised losses
        labels = labelled_b["labels"]
        multitask_labels = labelled_b["multitask_labels"]

        # pdb.set_trace()
        labelled_discourse_mask = labelled_b["span_attention_mask"].eq(1)  # (b, num_discourse)

        ce_logits = labelled_logits[:, :, :self.num_original_labels]
        multitask_logits = labelled_logits[:, :, self.num_original_labels:]

        logits = ce_logits
        ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

        if self.config["use_tsa"]:
            tsa_thresh = self.get_tsa_thresh(batch_idx)
            tsa_mask = torch.exp(-ce_loss) < tsa_thresh

            # update masks
            nd = labelled_discourse_mask.size(1)
            labelled_discourse_mask = torch.logical_and(labelled_discourse_mask, tsa_mask.reshape(-1, nd))
        labelled_multitask_mask = labelled_discourse_mask.unsqueeze(2).repeat(1, 1, 2)

        ce_loss = torch.masked_select(ce_loss.reshape(-1, 1), labelled_discourse_mask.reshape(-1, 1)).mean()
        loss_dict["ce_loss"] = ce_loss

        focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss = torch.masked_select(focal_loss.reshape(-1, 1), labelled_discourse_mask.reshape(-1, 1)).mean()
        loss_dict["focal_loss"] = focal_loss

        multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
        multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), labelled_multitask_mask.reshape(-1, 1)).mean()
        loss_dict["multitask_loss"] = multitask_loss

        loss = ce_loss + self.config["lambda_uda"]*uda_loss + multitask_loss + focal_loss

        return logits, loss, loss_dict

    def forward(self, labelled_b, original_unlabelled_b=None, augmented_unlabelled_b=None, is_train=True, batch_idx=0, **kwargs):
        if is_train:
            return self.train_step(labelled_b, original_unlabelled_b, augmented_unlabelled_b, batch_idx)

        else:  # validation
            labelled_logits = self.get_logits(labelled_b)
            logits = labelled_logits[:, :, :self.num_original_labels]
            return logits, None, dict()


#--------------------- Feedback Model for Meta Pseudo Labels ---------------------------#


class FeedbackModelMPL(nn.Module):
    """The feedback prize effectiveness baseline model with UDA
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelMPL, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.focal_loss_fn = FocalLossMPL(gamma=3.0, ignore_index=-1, reduction='none')
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def get_logits(self, batch):
        """obtain classification logits for the current batch
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        span_head_idxs = batch["span_head_idxs"]
        span_tail_idxs = batch["span_tail_idxs"]
        span_attention_mask = batch["span_attention_mask"]

        # batch size
        bs = input_ids.shape[0]

        # transformer output
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        # TODO: vectorize
        mean_feature_vector = []
        for i in range(bs):
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # feature_vector = mean_feature_vector
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        return logits

    def compute_loss(self, logits, labels, multitask_labels, span_attention_mask):
        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None

        main_mask = span_attention_mask  # .bool()
        multitask_mask = main_mask.unsqueeze(2).repeat(1, 1, 2)

        # split logits into ce_logits and bce_logits
        ce_logits = logits[:, :, :self.num_original_labels]
        multitask_logits = logits[:, :, self.num_original_labels:]

        multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
        multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_mask.reshape(-1, 1)).mean()
        loss_dict["multitask_loss"] = multitask_loss
        # assign ce_logits to logits to continue usual workflow
        logits = ce_logits

        ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))  # b, nd
        ce_loss = torch.masked_select(ce_loss.reshape(-1, 1), main_mask.reshape(-1, 1)).mean()

        focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss = torch.masked_select(focal_loss.reshape(-1, 1), main_mask.reshape(-1, 1)).mean()

        loss_dict["ce_loss"] = ce_loss
        loss = ce_loss + focal_loss + multitask_loss
        return loss, loss_dict


class FeedbackModelStudentFT(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelStudentFT, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # # RNN Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # # discourse type embeddings
        # self.num_dt_embeddings = 8
        # self.dt_embedding_dim = 16
        # self.dt_embeddings = nn.Embedding(self.num_dt_embeddings, self.dt_embedding_dim, padding_idx=0)
        # self.dt_embeddings.weight.data.normal_(mean=0.0, std=self.base_model.config.initializer_range)

        # self.fpe_dt_rnn = nn.GRU(
        #     input_size=self.dt_embedding_dim,
        #     hidden_size=self.dt_embedding_dim//2,
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.dt_layer_norm = LayerNorm(self.dt_embedding_dim, self.base_model.config.layer_norm_eps)

        # feature_size += self.dt_embedding_dim
        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=3.0, ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        discourse_type_ids,
        span_attention_mask,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

        mean_feature_vector = []

        for i in range(bs):  # TODO: vectorize
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # attend to other features
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # # discourse type sequence feature
        # dt_encodings = self.dt_embeddings(discourse_type_ids)  # (b, nd, h_dt)
        # dt_encodings = self.dt_layer_norm(dt_encodings)
        # self.fpe_dt_rnn.flatten_parameters()
        # dt_encodings = self.fpe_dt_rnn(dt_encodings)[0]  # LSTM layer outputs

        # feature_vector = torch.cat([feature_vector, dt_encodings], dim=-1)

        feature_vector = self.dropout(feature_vector)  # dropout

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None

        logits = self.classifier(feature_vector)
        # pdb.set_trace()

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits = logits[:, :, :self.num_original_labels]
            multitask_logits = logits[:, :, self.num_original_labels:]
            multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
            multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            loss_dict["multitask_loss"] = multitask_loss
            # assign ce_logits to logits to continue usual workflow
            logits = ce_logits

        ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss = self.focal_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))

        loss_dict["ce_loss"] = ce_loss

        if self.config["use_multitask"]:
            loss = ce_loss + focal_loss + multitask_loss
            # loss = ce_loss + multitask_loss
        else:
            loss = ce_loss + focal_loss
            # loss = ce_loss

        return logits, loss, loss_dict
#-------------------- Model with a classifier for each discourse type ------------------------------#


class FeedbackModelNeo(nn.Module):
    """The feedback prize effectiveness baseline model (will not be maintained)
    """

    def __init__(self, config):
        super(FeedbackModelNeo, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False})  # , "max_position_embeddings": 1024})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        print("=="*40)
        print("enabling gradient checkpointing...")
        self.base_model.gradient_checkpointing_enable()
        print("=="*40)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        if config["num_layers_reinit"] > 0:
            print("=="*40)
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # Classifier
        hidden_size = self.base_model.config.hidden_size
        self.num_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("=="*40)
            print("using multi-task approach...")
            self.num_original_labels = self.num_labels
            self.num_labels += self.config["num_additional_labels"]
            print("=="*40)

        num_discourse_types = 7
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, self.num_labels) for _ in range(num_discourse_types+1)])

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        discourse_type_ids,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]
        num_discourse = span_head_idxs.shape[1]

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
        )

        encoder_layer = outputs[0]

        mean_feature_vector = []
        for i in range(bs):
            vec_i = []
            for a, b in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, a+1:b], dim=0)  # [h]
                vec_i.append(tmp)
            vec_i = torch.stack(vec_i)  # (num_disourse, h)
            mean_feature_vector.append(vec_i)

        feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        feature_vector = self.dropout(feature_vector)

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None

        # logits = self.classifier(feature_vector)
        logits_list = [classifier(feature_vector) for classifier in self.classifiers]
        logits_list = torch.stack(logits_list, dim=2)  # (b, num_discourse, num_discourse_types, num_labels)
        # pdb.set_trace()
        # TODO: vectorize
        logits = []
        for i in range(bs):
            ex_logits = []
            for j in range(num_discourse):
                out = logits_list[i, j, discourse_type_ids[i, j]]  # [num_labels]
                ex_logits.append(out)
            ex_logits = torch.stack(ex_logits)  # [num_discourse, num_labels]
            logits.append(ex_logits)
        logits = torch.stack(logits)  # [b, num_discourse, num_labels]

        # logits = logits[torch.arange(bs).unsqueeze(-1), discourse_type_ids]

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits = logits[:, :, :self.num_original_labels]
            multitask_logits = logits[:, :, self.num_original_labels:]
            multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
            multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            loss_dict["multitask_loss"] = multitask_loss
            # assign ce_logits to logits to continue usual workflow
            logits = ce_logits

        ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
        loss_dict["ce_loss"] = ce_loss

        if self.config["use_multitask"]:
            loss = ce_loss + multitask_loss
        else:
            loss = ce_loss

        return logits, loss, loss_dict


class FeedbackModelEnb(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelEnb, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False})  # TODO: set "max_position_embeddings": 1024 in mlm and here
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # Width embeddings
        self.num_width_embeddings = 64
        self.width_embedding_dim = 8
        self.width_bucket_size = 16
        self.width_embeddings = nn.Embedding(self.num_width_embeddings, self.width_embedding_dim)
        self.width_embeddings.weight.data.normal_(mean=0.0, std=self.base_model.config.initializer_range)

        # Essay Length embeddings
        self.num_length_embeddings = 64
        self.length_embedding_dim = 8
        self.length_bucket_size = 32
        self.length_embeddings = nn.Embedding(self.num_length_embeddings, self.length_embedding_dim)
        self.length_embeddings.weight.data.normal_(mean=0.0, std=self.base_model.config.initializer_range)

        # classification
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size + self.width_embedding_dim + self.length_embedding_dim
        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        width_feature_vector, mean_feature_vector = [], []
        length_feature_vector = []
        lengths = attention_mask.sum(dim=1)  # (bs)

        for i in range(bs):  # TODO: vectorize
            span_width_vec_i, span_vec_i = [], []

            length_vec_i = []
            len_i = lengths[i]

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                # width feature
                w = torch.div(tail-head, self.width_bucket_size, rounding_mode="floor")
                w = torch.clamp(w, min=0, max=self.num_width_embeddings-1)
                span_width_vec_i.append(self.width_embeddings(w))

                # length feature
                lv = torch.div(len_i, self.length_bucket_size, rounding_mode="floor")
                lv = torch.clamp(lv, min=0, max=self.num_length_embeddings-1)
                length_vec_i.append(self.length_embeddings(lv))

                # span feature
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)

            span_width_vec_i = torch.stack(span_width_vec_i)
            length_vec_i = torch.stack(length_vec_i)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)

            width_feature_vector.append(span_width_vec_i)
            mean_feature_vector.append(span_vec_i)
            length_feature_vector.append(length_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        width_vector = torch.stack(width_feature_vector)  # (bs, num_discourse, wh)
        length_feature_vector = torch.stack(length_feature_vector)

        feature_vector = torch.cat([mean_feature_vector, width_vector, length_feature_vector],
                                   dim=-1)  # (bs, num_discourse, h + wh)
        feature_vector = self.dropout(feature_vector)

        # compute logits and loss
        loss_dict = dict()
        loss_dict["multitask_loss"] = None
        loss_dict["ce_loss"] = None

        logits = self.classifier(feature_vector)
        # pdb.set_trace()

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits = logits[:, :, :self.num_original_labels]
            multitask_logits = logits[:, :, self.num_original_labels:]
            multitask_loss = self.multitask_loss_fn(multitask_logits, multitask_labels)
            multitask_loss = torch.masked_select(multitask_loss.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            loss_dict["multitask_loss"] = multitask_loss
            # assign ce_logits to logits to continue usual workflow
            logits = ce_logits

        ce_loss = self.ce_loss_fn(logits.view(-1, self.num_original_labels), labels.view(-1))
        loss_dict["ce_loss"] = ce_loss

        if self.config["use_multitask"]:
            loss = ce_loss + multitask_loss
        else:
            loss = ce_loss

        return logits, loss, loss_dict
