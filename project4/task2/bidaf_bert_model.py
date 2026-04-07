"""
BiDAF with BERT Embeddings (BiDAF-BERT)

Replaces the word + character + highway embedding layers with frozen
bert-base-multilingual-uncased contextual embeddings, projected down
to match the BiDAF hidden dimension.  Everything from the contextual
BiLSTM onward is identical to the baseline BiDAF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from bidaf_model import (
    AttentionFlowLayer,
    ContextualEmbedding,
    ModelingLayer,
    OutputLayer,
)


class BiDAF_BERT(nn.Module):
    """
    BiDAF model that uses frozen BERT embeddings instead of
    GloVe + character CNN + highway network.

    Architecture:
        1. Frozen BERT encoder  ->  768-dim contextual embeddings
        2. Linear projection    ->  project 768 -> embed_dim
        3. Contextual BiLSTM    (same as baseline)
        4. Attention Flow       (same as baseline)
        5. Modeling BiLSTM      (same as baseline)
        6. Output Layer         (same as baseline)
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-multilingual-uncased",
        hidden_dim: int = 100,
        num_modeling_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Frozen BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        bert_dim = self.bert.config.hidden_size  # 768

        # 2. Project BERT output to BiDAF embedding dimension
        embed_dim = hidden_dim * 2  # match word_embed + char_embed size
        self.projection = nn.Linear(bert_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # 3. Contextual Embedding (BiLSTM)
        self.contextual = ContextualEmbedding(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        contextual_dim = hidden_dim * 2

        # 4. Attention Flow Layer
        self.attention = AttentionFlowLayer(hidden_dim=contextual_dim)
        attention_dim = contextual_dim * 4

        # 5. Modeling Layer
        self.modeling = ModelingLayer(
            input_dim=attention_dim,
            hidden_dim=hidden_dim,
            num_layers=num_modeling_layers,
            dropout=dropout,
        )
        modeling_dim = hidden_dim * 2

        # 6. Output Layer
        self.output = OutputLayer(
            hidden_dim=contextual_dim,
            modeling_dim=modeling_dim,
        )

    def _bert_embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run frozen BERT and project output."""
        with torch.no_grad():
            bert_out = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )
        # last_hidden_state: (batch, seq_len, 768)
        embeddings = bert_out.last_hidden_state
        # Project: (batch, seq_len, embed_dim)
        embeddings = self.proj_dropout(F.relu(self.projection(embeddings)))
        return embeddings

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_ids:  (batch, context_len) BERT token ids
            context_mask: (batch, context_len) attention mask
            query_ids:    (batch, query_len)   BERT token ids
            query_mask:   (batch, query_len)   attention mask

        Returns:
            start_logits: (batch, context_len)
            end_logits:   (batch, context_len)
        """
        # Embed with BERT + projection
        context_emb = self._bert_embed(context_ids, context_mask)
        query_emb = self._bert_embed(query_ids, query_mask)

        # Contextual BiLSTM
        context_encoded = self.contextual(context_emb, context_mask)
        query_encoded = self.contextual(query_emb, query_mask)

        # Attention flow
        g = self.attention(context_encoded, query_encoded, context_mask, query_mask)

        # Modeling
        m = self.modeling(g)

        # Output
        start_logits, end_logits = self.output(g, m, context_mask)
        return start_logits, end_logits

    def get_answer_span(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        return torch.argmax(start_probs, dim=-1), torch.argmax(end_probs, dim=-1)
