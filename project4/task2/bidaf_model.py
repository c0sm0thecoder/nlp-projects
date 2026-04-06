"""
BiDAF (Bidirectional Attention Flow) Model Implementation

Based on the paper: "Bidirectional Attention Flow for Machine Comprehension"
by Seo et al. (2017) - https://arxiv.org/abs/1611.01603

Architecture:
1. Character Embedding Layer (CNN)
2. Word Embedding Layer (GloVe/Word2Vec)
3. Highway Network
4. Contextual Embedding Layer (BiLSTM)
5. Attention Flow Layer (Context-to-Query & Query-to-Context)
6. Modeling Layer (BiLSTM)
7. Output Layer (Start/End position prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterEmbedding(nn.Module):
    """Character-level CNN embedding layer."""

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 8,
        num_filters: int = 100,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, word_len) - character indices

        Returns:
            (batch, seq_len, num_filters) - character-level embeddings
        """
        batch_size, seq_len, word_len = x.shape

        # Reshape for embedding: (batch * seq_len, word_len)
        x = x.view(-1, word_len)

        # Character embedding: (batch * seq_len, word_len, char_embed_dim)
        x = self.char_embed(x)
        x = self.dropout(x)

        # Conv1d expects (batch, channels, seq): (batch * seq_len, char_embed_dim, word_len)
        x = x.transpose(1, 2)

        # Apply convolution: (batch * seq_len, num_filters, word_len)
        x = self.conv(x)
        x = F.relu(x)

        # Max pooling over word length: (batch * seq_len, num_filters)
        x, _ = x.max(dim=2)

        # Reshape back: (batch, seq_len, num_filters)
        x = x.view(batch_size, seq_len, -1)

        return x


class Highway(nn.Module):
    """Highway Network layer for combining embeddings."""

    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        self.linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, input_dim)
        """
        for i in range(self.num_layers):
            h = F.relu(self.linear[i](x))
            g = torch.sigmoid(self.gate[i](x))
            x = g * h + (1 - g) * x

        return x


class ContextualEmbedding(nn.Module):
    """Contextual embedding layer using BiLSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - optional padding mask

        Returns:
            (batch, seq_len, hidden_dim * 2)
        """
        x = self.dropout(x)
        output, _ = self.lstm(x)
        return output


class AttentionFlowLayer(nn.Module):
    """
    Bidirectional Attention Flow layer.

    Computes:
    1. Context-to-Query (C2Q) attention
    2. Query-to-Context (Q2C) attention
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Similarity weight matrix: W_s * [h; u; h o u]
        # h: context, u: query, o: element-wise product
        self.w_sim = nn.Linear(hidden_dim * 3, 1, bias=False)

    def forward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        context_mask: torch.Tensor = None,
        query_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            context: (batch, context_len, hidden_dim) - H
            query: (batch, query_len, hidden_dim) - U
            context_mask: (batch, context_len)
            query_mask: (batch, query_len)

        Returns:
            G: (batch, context_len, hidden_dim * 4) - query-aware context representation
        """
        batch_size = context.size(0)
        context_len = context.size(1)
        query_len = query.size(1)
        hidden_dim = context.size(2)

        # Compute similarity matrix S(i,j) = w_s^T [h_i; u_j; h_i o u_j]
        # Shape: (batch, context_len, query_len)

        # Expand tensors for pairwise computation
        # context_expanded: (batch, context_len, query_len, hidden_dim)
        context_expanded = context.unsqueeze(2).expand(-1, -1, query_len, -1)
        # query_expanded: (batch, context_len, query_len, hidden_dim)
        query_expanded = query.unsqueeze(1).expand(-1, context_len, -1, -1)

        # Concatenate [h; u; h o u]: (batch, context_len, query_len, hidden_dim * 3)
        combined = torch.cat(
            [context_expanded, query_expanded, context_expanded * query_expanded],
            dim=-1,
        )

        # Similarity matrix: (batch, context_len, query_len)
        similarity = self.w_sim(combined).squeeze(-1)

        # Apply masks if provided
        if query_mask is not None:
            # Mask out padding in query
            query_mask_expanded = query_mask.unsqueeze(1).expand(-1, context_len, -1)
            similarity = similarity.masked_fill(~query_mask_expanded.bool(), float("-inf"))

        # Context-to-Query (C2Q) attention
        # For each context word, attend to query words
        # a_i = softmax(S_i:)
        c2q_attn = F.softmax(similarity, dim=-1)  # (batch, context_len, query_len)
        # U_tilde = sum_j(a_ij * U_j): (batch, context_len, hidden_dim)
        c2q = torch.bmm(c2q_attn, query)

        # Query-to-Context (Q2C) attention
        # Find most relevant context word for each query word
        # b = softmax(max_j(S_ij))
        q2c_attn = F.softmax(similarity.max(dim=-1)[0], dim=-1)  # (batch, context_len)
        # H_tilde = sum_i(b_i * H_i): (batch, hidden_dim)
        q2c = torch.bmm(q2c_attn.unsqueeze(1), context).squeeze(1)
        # Tile to match context length: (batch, context_len, hidden_dim)
        q2c = q2c.unsqueeze(1).expand(-1, context_len, -1)

        # Combine: G = [H; U_tilde; H o U_tilde; H o H_tilde]
        # Shape: (batch, context_len, hidden_dim * 4)
        g = torch.cat(
            [context, c2q, context * c2q, context * q2c],
            dim=-1,
        )

        return g


class ModelingLayer(nn.Module):
    """Modeling layer using stacked BiLSTMs."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, context_len, input_dim)

        Returns:
            (batch, context_len, hidden_dim * 2)
        """
        x = self.dropout(x)
        output, _ = self.lstm(x)
        return output


class OutputLayer(nn.Module):
    """Output layer for predicting start and end positions."""

    def __init__(self, hidden_dim: int, modeling_dim: int):
        super().__init__()
        # Start position: p1 = softmax(w_p1^T [G; M])
        self.w_start = nn.Linear(hidden_dim * 4 + modeling_dim, 1)
        # End position uses another BiLSTM
        self.end_lstm = nn.LSTM(
            input_size=modeling_dim,
            hidden_size=modeling_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        # End position: p2 = softmax(w_p2^T [G; M2])
        self.w_end = nn.Linear(hidden_dim * 4 + modeling_dim, 1)

    def forward(
        self,
        g: torch.Tensor,
        m: torch.Tensor,
        context_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: (batch, context_len, hidden_dim * 4) - attention output
            m: (batch, context_len, modeling_dim) - modeling layer output
            context_mask: (batch, context_len)

        Returns:
            start_logits: (batch, context_len)
            end_logits: (batch, context_len)
        """
        # Start position logits
        start_input = torch.cat([g, m], dim=-1)
        start_logits = self.w_start(start_input).squeeze(-1)

        # End position - pass M through another BiLSTM
        m2, _ = self.end_lstm(m)
        end_input = torch.cat([g, m2], dim=-1)
        end_logits = self.w_end(end_input).squeeze(-1)

        # Apply mask if provided
        if context_mask is not None:
            mask = ~context_mask.bool()
            start_logits = start_logits.masked_fill(mask, float("-inf"))
            end_logits = end_logits.masked_fill(mask, float("-inf"))

        return start_logits, end_logits


class BiDAF(nn.Module):
    """
    Bidirectional Attention Flow (BiDAF) model for Reading Comprehension.

    Takes a question and context passage as input, outputs start and end
    positions of the answer span within the context.
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        word_embed_dim: int = 100,
        char_embed_dim: int = 8,
        char_num_filters: int = 100,
        hidden_dim: int = 100,
        num_highway_layers: int = 2,
        num_modeling_layers: int = 2,
        dropout: float = 0.2,
        pretrained_word_embeddings: torch.Tensor = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 1. Word Embedding Layer
        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=0)
        if pretrained_word_embeddings is not None:
            self.word_embed.weight.data.copy_(pretrained_word_embeddings)
            self.word_embed.weight.requires_grad = False  # Freeze pretrained embeddings

        # 2. Character Embedding Layer
        self.char_embed = CharacterEmbedding(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            num_filters=char_num_filters,
        )

        # Combined embedding dimension
        embed_dim = word_embed_dim + char_num_filters

        # 3. Highway Network
        self.highway = Highway(input_dim=embed_dim, num_layers=num_highway_layers)

        # 4. Contextual Embedding Layer (BiLSTM)
        self.contextual = ContextualEmbedding(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        contextual_dim = hidden_dim * 2  # BiLSTM output

        # 5. Attention Flow Layer
        self.attention = AttentionFlowLayer(hidden_dim=contextual_dim)
        attention_dim = contextual_dim * 4  # [H; U_tilde; H*U_tilde; H*H_tilde]

        # 6. Modeling Layer
        self.modeling = ModelingLayer(
            input_dim=attention_dim,
            hidden_dim=hidden_dim,
            num_layers=num_modeling_layers,
            dropout=dropout,
        )
        modeling_dim = hidden_dim * 2  # BiLSTM output

        # 7. Output Layer
        self.output = OutputLayer(
            hidden_dim=contextual_dim,
            modeling_dim=modeling_dim,
        )

    def forward(
        self,
        context_word_ids: torch.Tensor,
        context_char_ids: torch.Tensor,
        query_word_ids: torch.Tensor,
        query_char_ids: torch.Tensor,
        context_mask: torch.Tensor = None,
        query_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_word_ids: (batch, context_len) - word indices for context
            context_char_ids: (batch, context_len, word_len) - char indices for context
            query_word_ids: (batch, query_len) - word indices for query
            query_char_ids: (batch, query_len, word_len) - char indices for query
            context_mask: (batch, context_len) - 1 for valid tokens, 0 for padding
            query_mask: (batch, query_len) - 1 for valid tokens, 0 for padding

        Returns:
            start_logits: (batch, context_len) - logits for start position
            end_logits: (batch, context_len) - logits for end position
        """
        # 1 & 2. Word and Character Embeddings
        context_word_emb = self.word_embed(context_word_ids)
        context_char_emb = self.char_embed(context_char_ids)
        context_emb = torch.cat([context_word_emb, context_char_emb], dim=-1)

        query_word_emb = self.word_embed(query_word_ids)
        query_char_emb = self.char_embed(query_char_ids)
        query_emb = torch.cat([query_word_emb, query_char_emb], dim=-1)

        # 3. Highway Network
        context_emb = self.highway(context_emb)
        query_emb = self.highway(query_emb)

        # 4. Contextual Embedding (BiLSTM)
        context_encoded = self.contextual(context_emb, context_mask)  # H
        query_encoded = self.contextual(query_emb, query_mask)  # U

        # 5. Attention Flow
        g = self.attention(context_encoded, query_encoded, context_mask, query_mask)

        # 6. Modeling Layer
        m = self.modeling(g)

        # 7. Output Layer
        start_logits, end_logits = self.output(g, m, context_mask)

        return start_logits, end_logits

    def get_answer_span(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the predicted answer span from logits.

        Args:
            start_logits: (batch, context_len)
            end_logits: (batch, context_len)

        Returns:
            start_positions: (batch,) - predicted start indices
            end_positions: (batch,) - predicted end indices
        """
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        start_positions = torch.argmax(start_probs, dim=-1)
        end_positions = torch.argmax(end_probs, dim=-1)

        return start_positions, end_positions


def compute_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for start and end positions.

    Args:
        start_logits: (batch, context_len)
        end_logits: (batch, context_len)
        start_positions: (batch,) - ground truth start indices
        end_positions: (batch,) - ground truth end indices

    Returns:
        loss: scalar tensor
    """
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)
    return (start_loss + end_loss) / 2
