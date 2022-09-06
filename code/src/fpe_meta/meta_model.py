import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention


class FeedbackMetaModelDev(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackMetaModelDev, self).__init__()

        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]

        print(f'Num fts: {self.num_meta_features}')
        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, 1e-7)

        self.meta_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # additional feature embeddings
        # topic embeddings
        num_topics = 16 + 1  # add 1 for padding
        embedding_dim = 8
        self.topic_embeddings = nn.Embedding(
            num_topics, embedding_dim, padding_idx=0
        )

        input_size = hidden_size + embedding_dim
        self.classifier = nn.Linear(input_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            meta_features,
            topic_nums,
            discourse_lens,
            essay_lens,
            num_discourse_elements,
            discourse_type_ids,
            labels=None,
            **kwargs
    ):

        # projection
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # run through rnn
        meta_features_rnn = self.meta_rnn(meta_features)[0]

        # dropout
        meta_features = meta_features + meta_features_rnn
        meta_features = self.dropout(meta_features)

        #
        topic_embeds = self.topic_embeddings(topic_nums)
        meta_features = torch.cat([meta_features, topic_embeds], axis=-1)

        logits = self.classifier(meta_features)

        # compute logits and loss
        loss = self.ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss

#########################################################################################################
#########################################################################################################


class FeedbackMetaModelResidual(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackMetaModelResidual, self).__init__()
        self.config = config

        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]
        # self.layer_norm_raw = LayerNorm(config["num_features"], 1e-7)

        print(f'Num fts: {self.num_meta_features}')
        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, 1e-7)

        self.meta_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            meta_features,
            attention_mask,
            span_attention_mask,
            discourse_type_ids,
            labels=None,
            **kwargs
    ):
        # meta_features = self.layer_norm_raw(meta_features)
        # projection
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # run through rnn
        meta_features_rnn = self.meta_rnn(meta_features)[0]

        # dropout
        meta_features = meta_features + meta_features_rnn
        meta_features = self.dropout(meta_features)
        logits = self.classifier(meta_features)

        # compute logits and loss
        if self.config["use_msd"]:
            feature_vector1 = self.dropout1(meta_features)
            feature_vector2 = self.dropout2(meta_features)
            feature_vector3 = self.dropout3(meta_features)
            feature_vector4 = self.dropout4(meta_features)
            feature_vector5 = self.dropout5(meta_features)

            logits1 = self.classifier(feature_vector1)
            logits2 = self.classifier(feature_vector2)
            logits3 = self.classifier(feature_vector3)
            logits4 = self.classifier(feature_vector4)
            logits5 = self.classifier(feature_vector5)
            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

            # compute logits and loss
            ce_loss1 = self.ce_loss_fn(logits1.view(-1, self.num_labels), labels.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, self.num_labels), labels.view(-1))
            ce_loss3 = self.ce_loss_fn(logits3.view(-1, self.num_labels), labels.view(-1))
            ce_loss4 = self.ce_loss_fn(logits4.view(-1, self.num_labels), labels.view(-1))
            ce_loss5 = self.ce_loss_fn(logits5.view(-1, self.num_labels), labels.view(-1))
            ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5) / 5
        else:
            ce_loss = self.ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        loss = ce_loss

        return logits, loss


class FeedbackMetaModel(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackMetaModel, self).__init__()

        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]

        print(f'Num fts: {self.num_meta_features}')
        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, 1e-7)

        self.meta_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # additional meta
        self.embedding_dim = 32
        self.num_embeddings = 16

        # Width embeddings
        self.width_bucket_size = 32
        self.width_embeddings = nn.Embedding(
            self.num_embeddings, self.embedding_dim
        )

        # Essay Length embeddings
        self.length_bucket_size = 64
        self.length_embeddings = nn.Embedding(
            self.num_embeddings, self.embedding_dim
        )

        # discourse type embeddings
        self.dt_embeddings = nn.Embedding(
            self.num_embeddings, self.embedding_dim, padding_idx=0
        )

        # topic embeddings
        self.topic_embeddings = nn.Embedding(
            self.num_embeddings, self.embedding_dim, padding_idx=0
        )

        # frequency embeddings
        self.frequency_embeddings = nn.Embedding(
            self.num_embeddings, self.embedding_dim)

        self.additional_projection = nn.Linear(5*self.embedding_dim, hidden_size)
        self.dense = nn.Linear(2*hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            meta_features,
            topic_nums,
            discourse_lens,
            essay_lens,
            num_discourse_elements,
            discourse_type_ids,
            labels=None,
            **kwargs
    ):
        # projection
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # run through rnn
        meta_features_rnn = self.meta_rnn(meta_features)[0]

        # dropout
        meta_features = meta_features + meta_features_rnn
        meta_features = self.dropout(meta_features)

        # extra features
        discourse_lens = torch.div(discourse_lens, self.width_bucket_size, rounding_mode="floor")
        discourse_lens = torch.clamp(discourse_lens, min=0, max=self.num_embeddings-1)

        essay_lens = torch.div(essay_lens, self.length_bucket_size, rounding_mode="floor")
        essay_lens = torch.clamp(essay_lens, min=0, max=self.num_embeddings-1)

        num_discourse_elements = torch.clamp(num_discourse_elements, min=0, max=self.num_embeddings-1)

        embeds_0 = self.width_embeddings(discourse_lens)
        embeds_1 = self.length_embeddings(essay_lens)
        embeds_2 = self.dt_embeddings(discourse_type_ids)
        embeds_3 = self.topic_embeddings(topic_nums)
        embeds_4 = self.frequency_embeddings(num_discourse_elements)   # (b, nd, h_dt)

        additional_features = torch.cat(
            [embeds_0, embeds_1, embeds_2, embeds_3, embeds_4], dim=-1
        )
        additional_features = self.additional_projection(additional_features)
        additional_features = self.dropout(additional_features)

        features = torch.cat([meta_features, additional_features], dim=-1)
        features = self.dropout(F.relu(self.dense(features)))
        logits = self.classifier(features)

        # compute logits and loss
        loss = self.ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss


class FeedbackMetaModelSimple(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackMetaModelSimple, self).__init__()

        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]

        print(f'Num fts: {self.num_meta_features}')
        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, 1e-7)

        self.meta_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            meta_features,
            attention_mask,
            span_attention_mask,
            discourse_type_ids,
            labels=None,
            **kwargs
    ):
        # projection
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # run through rnn
        meta_features = self.meta_rnn(meta_features)[0]

        # dropout
        meta_features = self.dropout(meta_features)
        logits = self.classifier(meta_features)

        # compute logits and loss
        loss = self.ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss


class FeedbackMetaModelLstmAttention(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("==" * 40)
        print("initializing the feedback model...")

        super(FeedbackMetaModelLstmAttention, self).__init__()

        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]

        print(f'Num fts: {self.num_meta_features}')
        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, 1e-7)

        self.meta_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": 16,
                "hidden_size": hidden_size,
                "attention_probs_dropout_prob": 0.1,
                "is_decoder": False,
            }
        )
        self.fpe_span_attention = BertAttention(
            attention_config,
            position_embedding_type="relative_key"
        )

        self.classifier = nn.Linear(hidden_size*2, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            meta_features,
            attention_mask,
            span_attention_mask,
            discourse_type_ids,
            labels=None,
            **kwargs
    ):
        # projection
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # run through rnn
        meta_features_rnn = self.meta_rnn(meta_features)[0]

        # attention
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        meta_features_attention = self.fpe_span_attention(meta_features, extended_span_attention_mask)[0]

        # dropout
        meta_features = torch.cat([meta_features_rnn, meta_features_attention], dim=-1)
        meta_features = self.dropout(meta_features)
        logits = self.classifier(meta_features)

        # compute logits and loss
        loss = self.ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss


class FeedbackMetaModelBertAttention(nn.Module):
    """
    The feedback prize effectiveness meta model for fast approach
    """

    def __init__(self, config):
        print("=="*40)
        print("initializing the feedback model...")

        super(FeedbackMetaModelBertAttention, self).__init__()
        self.config = config
        self.num_labels = config["num_labels"]
        self.num_meta_features = config["num_features"]

        # dropouts
        self.dropout = nn.Dropout(self.config["dropout"])
        hidden_size = 512
        self.projection = nn.Linear(self.num_meta_features, hidden_size)

        # multi-head attention over span representations
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": 8,
                "hidden_size": hidden_size,
                "attention_probs_dropout_prob": 0.1,
                "is_decoder": False,
            }
        )
        self.fpe_span_attention = BertAttention(
            attention_config,
            position_embedding_type="relative_key"
        )

        # classification
        self.layer_norm = LayerNorm(hidden_size, 1e-7)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Loss function
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        meta_features,
        attention_mask,
        span_attention_mask,
        discourse_type_ids,
        labels=None,
        **kwargs
    ):
        meta_features = self.projection(meta_features)

        # layer normalization
        meta_features = self.layer_norm(meta_features)

        # dropout
        meta_features = self.dropout(meta_features)

        # attention
        extended_span_attention_mask = span_attention_mask[:, None, None, :]
        # extended_span_attention_mask = extended_span_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        meta_features = self.fpe_span_attention(meta_features, extended_span_attention_mask)[0]
        meta_features = self.dropout(meta_features)
        logits = self.classifier(meta_features)

        # compute logits and loss
        loss = self.ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss
