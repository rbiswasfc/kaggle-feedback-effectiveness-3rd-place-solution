import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel
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
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": config["max_position_embeddings"]})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # re-initialization
        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # freeze embeddings
        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
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

        # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(feature_size, self.num_labels)

        # dropout family
        self.dropout = nn.Dropout(self.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=config["focal_gamma"], ignore_index=-1)
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

        # batch size
        bs = input_ids.shape[0]

        # get contextual token representations from base transformer
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        # run contextual information through lstm
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        # extract span representation # TODO: vectorize
        mean_feature_vector = []
        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # apply multi-head attention over span representation
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # apply multi-sample dropouts
        # feature_vector = self.dropout(feature_vector)
        feature_vector1 = self.dropout1(feature_vector)
        feature_vector2 = self.dropout2(feature_vector)
        feature_vector3 = self.dropout3(feature_vector)
        feature_vector4 = self.dropout4(feature_vector)
        feature_vector5 = self.dropout5(feature_vector)

        # logits = self.classifier(feature_vector)

        logits1 = self.classifier(feature_vector1)
        logits2 = self.classifier(feature_vector2)
        logits3 = self.classifier(feature_vector3)
        logits4 = self.classifier(feature_vector4)
        logits5 = self.classifier(feature_vector5)

        logits = (logits1 + logits2 + logits3 + logits4 + logits5)/5

        # compute loss
        loss_dict = dict()
        loss_dict["ce_loss"] = None
        loss_dict["focal_loss"] = None
        loss_dict["multitask_loss"] = None
        loss = None

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits1 = logits1[:, :, :self.num_original_labels]
            ce_logits2 = logits2[:, :, :self.num_original_labels]
            ce_logits3 = logits3[:, :, :self.num_original_labels]
            ce_logits4 = logits4[:, :, :self.num_original_labels]
            ce_logits5 = logits5[:, :, :self.num_original_labels]
            ce_logits = (ce_logits1 + ce_logits2 + ce_logits3 + ce_logits4 + ce_logits5)/5

            multitask_logits1 = logits1[:, :, self.num_original_labels:]
            multitask_logits2 = logits2[:, :, self.num_original_labels:]
            multitask_logits3 = logits3[:, :, self.num_original_labels:]
            multitask_logits4 = logits4[:, :, self.num_original_labels:]
            multitask_logits5 = logits5[:, :, self.num_original_labels:]

            multitask_loss1 = self.multitask_loss_fn(multitask_logits1, multitask_labels)
            multitask_loss1 = torch.masked_select(multitask_loss1.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss2 = self.multitask_loss_fn(multitask_logits2, multitask_labels)
            multitask_loss2 = torch.masked_select(multitask_loss2.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss3 = self.multitask_loss_fn(multitask_logits3, multitask_labels)
            multitask_loss3 = torch.masked_select(multitask_loss3.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss4 = self.multitask_loss_fn(multitask_logits4, multitask_labels)
            multitask_loss4 = torch.masked_select(multitask_loss4.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss5 = self.multitask_loss_fn(multitask_logits5, multitask_labels)
            multitask_loss5 = torch.masked_select(multitask_loss5.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss = (multitask_loss1 + multitask_loss2 + multitask_loss3 + multitask_loss4 + multitask_loss5)/5
            loss_dict["multitask_loss"] = multitask_loss

            # assign ce_logits to logits to continue usual workflow
            logits1 = ce_logits1
            logits2 = ce_logits2
            logits3 = ce_logits3
            logits4 = ce_logits4
            logits5 = ce_logits5
            logits = ce_logits

        ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5)/5

        focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5)/5

        loss_dict["ce_loss"] = ce_loss
        loss_dict["focal_loss"] = focal_loss

        if self.config["use_multitask"]:
            loss = ce_loss + multitask_loss + focal_loss
            # loss = ce_loss + multitask_loss
        else:
            loss = ce_loss + focal_loss
            # loss = ce_loss

        return logits, loss, loss_dict

#--------------- Feedback Model with Weighted Loss ----------------#


class FeedbackModelWeightedLoss(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelWeightedLoss, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": config["max_position_embeddings"]})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # re-initialization
        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # freeze embeddings
        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
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

        # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(feature_size, self.num_labels)

        # dropout family
        self.dropout = nn.Dropout(self.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.focal_loss_fn = FocalLoss(gamma=config["focal_gamma"], ignore_index=-1)
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
        weights_factor=None,
        **kwargs
    ):

        # batch size
        bs = input_ids.shape[0]

        # get contextual token representations from base transformer
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        # run contextual information through lstm
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        # extract span representation # TODO: vectorize
        mean_feature_vector = []
        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # apply multi-head attention over span representation
        backup_span_attention_mask = span_attention_mask.clone()

        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)

        # apply multi-sample dropouts
        # feature_vector = self.dropout(feature_vector)
        feature_vector1 = self.dropout1(feature_vector)
        feature_vector2 = self.dropout2(feature_vector)
        feature_vector3 = self.dropout3(feature_vector)
        feature_vector4 = self.dropout4(feature_vector)
        feature_vector5 = self.dropout5(feature_vector)

        # logits = self.classifier(feature_vector)

        logits1 = self.classifier(feature_vector1)
        logits2 = self.classifier(feature_vector2)
        logits3 = self.classifier(feature_vector3)
        logits4 = self.classifier(feature_vector4)
        logits5 = self.classifier(feature_vector5)

        logits = (logits1 + logits2 + logits3 + logits4 + logits5)/5

        # compute loss
        loss_dict = dict()
        loss_dict["ce_loss"] = None
        loss_dict["focal_loss"] = None
        loss_dict["multitask_loss"] = None
        loss = None

        if self.config["use_multitask"]:
            # split logits into ce_logits and bce_logits
            ce_logits1 = logits1[:, :, :self.num_original_labels]
            ce_logits2 = logits2[:, :, :self.num_original_labels]
            ce_logits3 = logits3[:, :, :self.num_original_labels]
            ce_logits4 = logits4[:, :, :self.num_original_labels]
            ce_logits5 = logits5[:, :, :self.num_original_labels]
            ce_logits = (ce_logits1 + ce_logits2 + ce_logits3 + ce_logits4 + ce_logits5)/5

            multitask_logits1 = logits1[:, :, self.num_original_labels:]
            multitask_logits2 = logits2[:, :, self.num_original_labels:]
            multitask_logits3 = logits3[:, :, self.num_original_labels:]
            multitask_logits4 = logits4[:, :, self.num_original_labels:]
            multitask_logits5 = logits5[:, :, self.num_original_labels:]

            multitask_loss1 = self.multitask_loss_fn(multitask_logits1, multitask_labels)
            multitask_loss1 = torch.masked_select(multitask_loss1.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss2 = self.multitask_loss_fn(multitask_logits2, multitask_labels)
            multitask_loss2 = torch.masked_select(multitask_loss2.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss3 = self.multitask_loss_fn(multitask_logits3, multitask_labels)
            multitask_loss3 = torch.masked_select(multitask_loss3.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss4 = self.multitask_loss_fn(multitask_logits4, multitask_labels)
            multitask_loss4 = torch.masked_select(multitask_loss4.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss5 = self.multitask_loss_fn(multitask_logits5, multitask_labels)
            multitask_loss5 = torch.masked_select(multitask_loss5.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

            multitask_loss = (multitask_loss1 + multitask_loss2 + multitask_loss3 + multitask_loss4 + multitask_loss5)/5
            loss_dict["multitask_loss"] = multitask_loss

            # assign ce_logits to logits to continue usual workflow
            logits1 = ce_logits1
            logits2 = ce_logits2
            logits3 = ce_logits3
            logits4 = ce_logits4
            logits5 = ce_logits5
            logits = ce_logits

        ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))
        ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5)/5

        # apply weights and reduce
        # pdb.set_trace()
        mask = backup_span_attention_mask.eq(1)
        ce_loss = ce_loss*weights_factor.reshape(-1)
        ce_loss = torch.masked_select(ce_loss.reshape(-1, 1), mask.reshape(-1, 1)).mean()

        focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))
        focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5)/5

        loss_dict["ce_loss"] = ce_loss
        loss_dict["focal_loss"] = focal_loss

        if self.config["use_multitask"]:
            loss = ce_loss + multitask_loss + focal_loss
            # loss = ce_loss + multitask_loss
        else:
            loss = ce_loss + focal_loss
            # loss = ce_loss

        return logits, loss, loss_dict

#--------------- UDA ----------------------------------------------#


class FeedbackModelUDA(nn.Module):
    """The feedback prize effectiveness baseline model with UDA
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model for UDA...")

        super(FeedbackModelUDA, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": config["max_position_embeddings"]})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # re-initialization
        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # freeze embeddings
        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
            print("=="*40)

        # multi-head attention
        attention_config = deepcopy(self.base_model.config)
        attention_config.update({"relative_attention": False})
        self.fpe_span_attention = DebertaV2Attention(attention_config)

        # classification
        self.num_labels = self.config["num_labels"]
        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size

        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # dropouts
        self.dropout = StableDropout(self.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

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
        logits = self.classifier(feature_vector)
        return logits, feature_vector

    def train_step(self, labelled_b, original_unlabelled_b, augmented_unlabelled_b, batch_idx):
        # initialize losses
        loss_dict = dict()

        # get labelled logits
        _, labelled_features = self.get_logits(labelled_b)

        #------------------ UDA Loss ---------------------------------------------------#
        unlabelled_discourse_mask = original_unlabelled_b["span_attention_mask"].float()  # (b, num_discourse)

        with torch.no_grad():
            uw_logits, _ = self.get_logits(original_unlabelled_b)  # (b, num_discourse, 3)
            uw_probs = F.softmax(uw_logits, dim=-1)  # (b*num_discourse, 3) # KLdiv target
            confidence_mask = torch.max(uw_probs, dim=-1)[0].ge(self.config["confidence_threshold"]).float()

        us_logits, _ = self.get_logits(augmented_unlabelled_b)
        us_probs = F.log_softmax(us_logits/self.config["temperature"], dim=-1)  # (b, num_discourse, 3)

        uda_criterion = nn.KLDivLoss(reduction='none')
        uda_loss = torch.sum(torch.sum(uda_criterion(us_probs, uw_probs), dim=-1)
                             * confidence_mask * unlabelled_discourse_mask)
        uda_loss = uda_loss / torch.sum(confidence_mask*unlabelled_discourse_mask).clamp(min=1.0)
        loss_dict["uda_loss"] = uda_loss

        #------------------ Supervised Loss ------------------------------------------#
        labels = labelled_b["labels"]

        feature_vector1 = self.dropout1(labelled_features)
        feature_vector2 = self.dropout2(labelled_features)
        feature_vector3 = self.dropout3(labelled_features)
        feature_vector4 = self.dropout4(labelled_features)
        feature_vector5 = self.dropout5(labelled_features)

        logits1 = self.classifier(feature_vector1)
        logits2 = self.classifier(feature_vector2)
        logits3 = self.classifier(feature_vector3)
        logits4 = self.classifier(feature_vector4)
        logits5 = self.classifier(feature_vector5)
        logits = (logits1 + logits2 + logits3 + logits4 + logits5)/5

        ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_labels), labels.view(-1))
        ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_labels), labels.view(-1))
        ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_labels), labels.view(-1))
        ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_labels), labels.view(-1))
        ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_labels), labels.view(-1))
        ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5)/5
        loss_dict["ce_loss"] = ce_loss

        #------------------ Total Loss ------------------------------------------#
        loss = ce_loss + self.config["lambda_uda"]*uda_loss

        return logits, loss, loss_dict

    def forward(self, labelled_b, original_unlabelled_b=None, augmented_unlabelled_b=None, is_train=True, batch_idx=0, **kwargs):
        if is_train:
            return self.train_step(labelled_b, original_unlabelled_b, augmented_unlabelled_b, batch_idx)

        else:  # validation
            logits, _ = self.get_logits(labelled_b)
            return logits, None, dict()


#--------------- Knowledge Distillation ---------------------------#


class FeedbackModelKD(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelKD, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update({"add_pooling_layer": False, "max_position_embeddings": config["max_position_embeddings"]})
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # re-initialization
        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_deberta(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # freeze embeddings
        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
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

        # LSTM Head
        self.fpe_lstm_layer = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(feature_size, self.num_labels)

        # dropout family
        self.dropout = nn.Dropout(self.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=config["focal_gamma"], ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.kd_loss = nn.KLDivLoss(reduction="none")

    def get_logits(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        **kwargs,
    ):
        # batch size
        bs = input_ids.shape[0]

        # get contextual token representations from base transformer
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        encoder_layer = outputs[0]

        # run contextual information through lstm
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        # extract span representation # TODO: vectorize
        mean_feature_vector = []
        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_disourse, h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)  # (bs, num_disourse, h)
        mean_feature_vector = self.layer_norm(mean_feature_vector)

        # apply multi-head attention over span representation
        extended_span_attention_mask = span_attention_mask.unsqueeze(1).unsqueeze(2)
        span_attention_mask = extended_span_attention_mask * extended_span_attention_mask.squeeze(-2).unsqueeze(-1)
        span_attention_mask = span_attention_mask.byte()
        feature_vector = self.fpe_span_attention(mean_feature_vector, span_attention_mask)
        logits = self.classifier(feature_vector)
        return logits, feature_vector

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        labels=None,
        multitask_labels=None,
        teacher=None,
        **kwargs
    ):
        logits, feature_vector = self.get_logits(
            input_ids,
            attention_mask,
            span_head_idxs,
            span_tail_idxs,
            span_attention_mask
        )

        # feature_vector = self.dropout(feature_vector)
        feature_vector1 = self.dropout1(feature_vector)
        feature_vector2 = self.dropout2(feature_vector)
        feature_vector3 = self.dropout3(feature_vector)
        feature_vector4 = self.dropout4(feature_vector)
        feature_vector5 = self.dropout5(feature_vector)

        # logits = self.classifier(feature_vector)

        logits1 = self.classifier(feature_vector1)
        logits2 = self.classifier(feature_vector2)
        logits3 = self.classifier(feature_vector3)
        logits4 = self.classifier(feature_vector4)
        logits5 = self.classifier(feature_vector5)

        # compute loss
        loss = None
        loss_dict = dict()
        loss_dict["ce_loss"] = None
        loss_dict["kd_loss"] = None
        loss_dict["focal_loss"] = None
        loss_dict["multitask_loss"] = None

        if labels is not None:
            if self.config["use_multitask"]:
                # split logits into ce_logits and bce_logits
                ce_logits1 = logits1[:, :, :self.num_original_labels]
                ce_logits2 = logits2[:, :, :self.num_original_labels]
                ce_logits3 = logits3[:, :, :self.num_original_labels]
                ce_logits4 = logits4[:, :, :self.num_original_labels]
                ce_logits5 = logits5[:, :, :self.num_original_labels]
                ce_logits = (ce_logits1 + ce_logits2 + ce_logits3 + ce_logits4 + ce_logits5)/5

                multitask_logits1 = logits1[:, :, self.num_original_labels:]
                multitask_logits2 = logits2[:, :, self.num_original_labels:]
                multitask_logits3 = logits3[:, :, self.num_original_labels:]
                multitask_logits4 = logits4[:, :, self.num_original_labels:]
                multitask_logits5 = logits5[:, :, self.num_original_labels:]

                multitask_loss1 = self.multitask_loss_fn(multitask_logits1, multitask_labels)
                multitask_loss1 = torch.masked_select(multitask_loss1.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                multitask_loss2 = self.multitask_loss_fn(multitask_logits2, multitask_labels)
                multitask_loss2 = torch.masked_select(multitask_loss2.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                multitask_loss3 = self.multitask_loss_fn(multitask_logits3, multitask_labels)
                multitask_loss3 = torch.masked_select(multitask_loss3.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                multitask_loss4 = self.multitask_loss_fn(multitask_logits4, multitask_labels)
                multitask_loss4 = torch.masked_select(multitask_loss4.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                multitask_loss5 = self.multitask_loss_fn(multitask_logits5, multitask_labels)
                multitask_loss5 = torch.masked_select(multitask_loss5.reshape(-1, 1), multitask_labels.view(-1, 1) > -1).mean()

                multitask_loss = (multitask_loss1 + multitask_loss2 + multitask_loss3 + multitask_loss4 + multitask_loss5)/5
                loss_dict["multitask_loss"] = multitask_loss

                # assign ce_logits to logits to continue usual workflow
                logits1 = ce_logits1
                logits2 = ce_logits2
                logits3 = ce_logits3
                logits4 = ce_logits4
                logits5 = ce_logits5
                logits = ce_logits

            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))
            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5)/5

            focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_original_labels), labels.view(-1))
            focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5)/5

            loss_dict["ce_loss"] = ce_loss
            loss_dict["focal_loss"] = focal_loss

            #--------------- Knowledge Distillation ----------------------------------------------------#
            # compute distillation loss
            teacher.eval()
            with torch.no_grad():
                t_logits, t_features = teacher.get_logits(
                    input_ids,
                    attention_mask,
                    span_head_idxs,
                    span_tail_idxs,
                    span_attention_mask
                )
                t_logits = t_logits[:, :, :self.num_original_labels]
                kld_targets = F.softmax(t_logits/self.config["temperature"], dim=-1)

            kld_inputs1 = F.log_softmax(logits1/self.config["temperature"], dim=-1)
            kld_inputs2 = F.log_softmax(logits2/self.config["temperature"], dim=-1)
            kld_inputs3 = F.log_softmax(logits3/self.config["temperature"], dim=-1)
            kld_inputs4 = F.log_softmax(logits4/self.config["temperature"], dim=-1)
            kld_inputs5 = F.log_softmax(logits5/self.config["temperature"], dim=-1)

            kld_inputs = [kld_inputs1, kld_inputs2, kld_inputs3, kld_inputs4, kld_inputs5]

            for idx, kld_inputs_i in enumerate(kld_inputs):
                kd_loss_i = torch.sum(self.kd_loss(kld_inputs_i, kld_targets), dim=-1)
                kd_loss_i = torch.sum(kd_loss_i*span_attention_mask.float())
                kd_loss_i = kd_loss_i / torch.sum(span_attention_mask.float()).clamp(min=1.0)
                kd_loss_i = kd_loss_i * self.config["temperature"] ** 2
                if idx == 0:
                    kd_loss = kd_loss_i
                else:
                    kd_loss += kd_loss_i
            kd_loss = kd_loss/len(kld_inputs)
            loss_dict["kd_loss"] = kd_loss

            if self.config["use_multitask"]:
                loss = (1.0-self.config["alpha"])*(ce_loss + multitask_loss + focal_loss) + self.config["alpha"]*kd_loss
            else:
                loss = (1.0-self.config["alpha"])*(ce_loss + focal_loss) + self.config["alpha"]*kd_loss

        if logits.shape[-1] != self.num_original_labels:
            logits = logits[:, :, :self.num_original_labels]  # main logits

        return logits, loss, loss_dict
