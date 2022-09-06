import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertEncoder

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


def reinit_luke(base_model, num_reinit_layers):
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            """Initialize the weights"""
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                if module.embedding_dim == 1:  # embedding for bias parameters
                    module.weight.data.zero_()
                else:
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
        base_config.update(
            {"max_position_embeddings": config["max_position_embeddings"]+2}
        )
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
            reinit_luke(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # freeze embeddings
        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
            print("=="*40)

        self.num_labels = self.config["num_labels"]

        # LSTM Head
        hidden_size = self.base_model.config.hidden_size

        self.fpe_lstm_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # classification
        feature_size = hidden_size * 3
        self.classifier = nn.Linear(feature_size, self.num_labels)
        self.discourse_classifier = nn.Linear(feature_size, 7)  # 7 discourse elements

        # dropout family
        self.dropout = nn.Dropout(self.config["dropout"])
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=config["focal_gamma"], ignore_index=-1)

    def forward(
        self,
        input_ids,
        attention_mask,
        entity_ids,
        entity_attention_mask,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
        discourse_type_ids,
        labels=None,
        **kwargs
    ):

        # get contextual token representations from base transformer
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )

        # run contextual information through lstm
        encoder_layer = outputs.last_hidden_state
        encoder_layer_entity = outputs.entity_last_hidden_state

        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        hidden_size = outputs.last_hidden_state.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        start_states = torch.gather(encoder_layer, -2, entity_start_positions)
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        end_states = torch.gather(encoder_layer, -2, entity_end_positions)
        feature_vector = torch.cat([start_states, end_states, encoder_layer_entity],
                                   dim=2)  # check if should use lstm

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

        discourse_logits1 = self.discourse_classifier(feature_vector1)
        discourse_logits2 = self.discourse_classifier(feature_vector2)
        discourse_logits3 = self.discourse_classifier(feature_vector3)
        discourse_logits4 = self.discourse_classifier(feature_vector4)
        discourse_logits5 = self.discourse_classifier(feature_vector5)

        # compute loss
        loss_dict = dict()
        loss = None

        if labels is not None:

            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_labels), labels.view(-1))
            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5)/5

            focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_labels), labels.view(-1))
            focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_labels), labels.view(-1))
            focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_labels), labels.view(-1))
            focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_labels), labels.view(-1))
            focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_labels), labels.view(-1))
            focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5)/5

            ce_loss_discourse1 = self.ce_loss_fn(discourse_logits1.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse2 = self.ce_loss_fn(discourse_logits2.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse3 = self.ce_loss_fn(discourse_logits3.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse4 = self.ce_loss_fn(discourse_logits4.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse5 = self.ce_loss_fn(discourse_logits5.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse = (ce_loss_discourse1 + ce_loss_discourse2 +
                                 ce_loss_discourse3 + ce_loss_discourse4 + ce_loss_discourse5)/5

            loss_dict["ce_loss"] = ce_loss
            loss_dict["focal_loss"] = focal_loss
            loss_dict["multitask_loss"] = ce_loss_discourse

            loss = ce_loss + focal_loss + ce_loss_discourse

        return logits, loss, loss_dict


class FeedbackModelV2(nn.Module):
    """
    The feedback prize effectiveness model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackModelV2, self).__init__()
        self.config = config

        # base transformer
        base_config = AutoConfig.from_pretrained(self.config["base_model_path"])
        base_config.update(
            {"max_position_embeddings": config["max_position_embeddings"]+2}
        )
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
            reinit_luke(self.base_model, self.config["num_layers_reinit"])
            print("=="*40)

        # freeze embeddings
        if config["n_freeze"] > 0:
            print("=="*40)
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:config["n_freeze"]].requires_grad_(False)
            print("=="*40)

        self.num_labels = self.config["num_labels"]

        # LSTM Head
        hidden_size = self.base_model.config.hidden_size

        self.fpe_lstm_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fpe_lstm_layer_entity = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # classification
        feature_size = hidden_size * 3
        self.classifier = nn.Linear(feature_size, self.num_labels)
        self.discourse_classifier = nn.Linear(feature_size, 7)  # 7 discourse elements

        # dropout family
        self.dropout = nn.Dropout(self.config["dropout"])
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        # Bert Attention
        self.layer_norm = LayerNorm(feature_size, self.base_model.config.layer_norm_eps)

        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": self.base_model.config.num_attention_heads*3,
                "hidden_size": self.base_model.config.hidden_size*3,
                "attention_probs_dropout_prob": self.base_model.config.attention_probs_dropout_prob,
                "is_decoder": False,

            }
        )
        self.fpe_span_attention = BertAttention(attention_config, position_embedding_type="relative_key")

        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal_loss_fn = FocalLoss(gamma=config["focal_gamma"], ignore_index=-1)

    def forward(
        self,
        input_ids,
        attention_mask,
        entity_ids,
        entity_attention_mask,
        entity_position_ids,
        entity_start_positions,
        entity_end_positions,
        discourse_type_ids,
        labels=None,
        **kwargs
    ):

        # get contextual token representations from base transformer
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )

        # run contextual information through lstm
        encoder_layer = outputs.last_hidden_state
        encoder_layer_entity = outputs.entity_last_hidden_state

        # run through lstm layers
        self.fpe_lstm_layer.flatten_parameters()
        encoder_layer = self.fpe_lstm_layer(encoder_layer)[0]

        self.fpe_lstm_layer_entity.flatten_parameters()
        encoder_layer_entity = self.fpe_lstm_layer_entity(encoder_layer_entity)[0]

        hidden_size = outputs.last_hidden_state.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        start_states = torch.gather(encoder_layer, -2, entity_start_positions)
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        end_states = torch.gather(encoder_layer, -2, entity_end_positions)
        feature_vector = torch.cat([start_states, end_states, encoder_layer_entity],
                                   dim=2)  # check if should use lstm

        # # attention mechanism
        bert_feature_vector = self.layer_norm(feature_vector)
        extended_span_attention_mask = entity_attention_mask[:, None, None, :]
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        bert_feature_vector = self.fpe_span_attention(bert_feature_vector, extended_span_attention_mask)[0]
        bert_feature_vector = self.dropout(bert_feature_vector)

        # have bert encoder over features
        feature_vector = bert_feature_vector  # assign

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

        discourse_logits1 = self.discourse_classifier(feature_vector1)
        discourse_logits2 = self.discourse_classifier(feature_vector2)
        discourse_logits3 = self.discourse_classifier(feature_vector3)
        discourse_logits4 = self.discourse_classifier(feature_vector4)
        discourse_logits5 = self.discourse_classifier(feature_vector5)

        # compute loss
        loss_dict = dict()
        loss = None

        if labels is not None:

            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_labels), labels.view(-1))
            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5)/5

            focal_loss1 = self.focal_loss_fn(logits1.view(-1, self.num_labels), labels.view(-1))
            focal_loss2 = self.focal_loss_fn(logits2.view(-1, self.num_labels), labels.view(-1))
            focal_loss3 = self.focal_loss_fn(logits3.view(-1, self.num_labels), labels.view(-1))
            focal_loss4 = self.focal_loss_fn(logits4.view(-1, self.num_labels), labels.view(-1))
            focal_loss5 = self.focal_loss_fn(logits5.view(-1, self.num_labels), labels.view(-1))
            focal_loss = (focal_loss1 + focal_loss2 + focal_loss3 + focal_loss4 + focal_loss5)/5

            ce_loss_discourse1 = self.ce_loss_fn(discourse_logits1.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse2 = self.ce_loss_fn(discourse_logits2.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse3 = self.ce_loss_fn(discourse_logits3.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse4 = self.ce_loss_fn(discourse_logits4.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse5 = self.ce_loss_fn(discourse_logits5.view(-1, 7), discourse_type_ids.view(-1))
            ce_loss_discourse = (ce_loss_discourse1 + ce_loss_discourse2 +
                                 ce_loss_discourse3 + ce_loss_discourse4 + ce_loss_discourse5)/5

            loss_dict["ce_loss"] = ce_loss
            loss_dict["focal_loss"] = focal_loss
            loss_dict["multitask_loss"] = ce_loss_discourse

            loss = ce_loss + focal_loss + ce_loss_discourse

        return logits, loss, loss_dict
