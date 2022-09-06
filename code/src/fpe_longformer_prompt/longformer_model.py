import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.longformer.modeling_longformer import \
    LongformerAttention

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

#-------- Re-initialization ------------------------------------------------------#


def reinit_longformer(base_model, num_reinit_layers):
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            """Initialize the weights"""
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


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
        self.base_model = AutoModel.from_pretrained(self.config["base_model_path"], config=base_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.base_model.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])

        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit'] } layers of the base model...")
            reinit_longformer(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        self.num_labels = self.num_original_labels = self.config["num_labels"]

        if self.config["use_multitask"]:
            print("using multi-task approach...")
            self.num_labels += self.config["num_additional_labels"]

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": self.base_model.config.num_attention_heads,
                "hidden_size": self.base_model.config.hidden_size,
                "attention_probs_dropout_prob": self.base_model.config.attention_probs_dropout_prob,
                "is_decoder": False,

            }
        )
        self.fpe_span_attention = BertAttention(attention_config, position_embedding_type="relative_key")

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

        self.classifier = nn.Linear(feature_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=2.5, ignore_index=-1)
        self.multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        global_attention_mask,
        labels=None,
        multitask_labels=None,
        **kwargs
    ):

        bs = input_ids.shape[0]  # batch size

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            global_attention_mask=global_attention_mask,
        )
        encoder_layer = outputs[0]

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]  # LSTM layer outputs

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

        # attention mechanism
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        feature_vector = self.fpe_span_attention(mean_feature_vector, extended_span_attention_mask)[0]

        feature_vector = self.dropout(mean_feature_vector)
        logits = self.classifier(feature_vector)

        # compute logits and loss
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
