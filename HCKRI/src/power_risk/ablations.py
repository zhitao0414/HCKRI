from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class AblationRiskNet(nn.Module):
    """Configurable version of the proposed model for module ablation."""

    def __init__(
        self,
        feature_dim: int,
        n_sources: int,
        n_classes: int,
        config: ModelConfig,
        use_cnn: bool = True,
        use_lstm: bool = True,
        use_transformer: bool = True,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer
        self.use_gate = use_gate

        self.source_embedding = nn.Embedding(n_sources, config.source_embedding_dim)
        self.input_projection = nn.Linear(
            feature_dim + config.source_embedding_dim,
            config.model_dim,
        )

        if use_cnn:
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        config.model_dim,
                        config.model_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                    for kernel_size in config.cnn_kernel_sizes
                ]
            )
            self.cnn_fusion = nn.Linear(
                config.model_dim * len(config.cnn_kernel_sizes),
                config.model_dim,
            )
            self.cnn_norm = nn.LayerNorm(config.model_dim)
            self.dropout = nn.Dropout(config.dropout)

        if use_lstm:
            self.bilstm = nn.LSTM(
                input_size=config.model_dim,
                hidden_size=config.lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.lstm_projection = nn.Linear(2 * config.lstm_hidden, config.model_dim)
            self.temporal_norm = nn.LayerNorm(config.model_dim)

        self.source_projection = nn.Linear(config.source_embedding_dim, config.model_dim)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(config.model_dim * 2, config.model_dim),
                nn.Sigmoid(),
            )
        else:
            self.simple_fusion_norm = nn.LayerNorm(config.model_dim)

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.transformer_heads,
                dim_feedforward=config.model_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.transformer_layers,
            )
            self.position_projection = nn.Linear(1, config.model_dim)
            self.transformer_norm = nn.LayerNorm(config.model_dim)

        self.pool_attention = nn.Sequential(
            nn.Linear(config.model_dim, max(config.model_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(config.model_dim // 2, 1), 1),
        )
        self.stage_fusion = nn.Sequential(
            nn.LayerNorm(config.model_dim * 3),
            nn.Linear(config.model_dim * 3, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.embedding_norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_emb = self.source_embedding(source_ids)
        h = torch.cat([x, source_emb], dim=-1)
        h = self.input_projection(h)
        zero_embedding = torch.zeros(
            h.shape[0],
            h.shape[-1],
            device=h.device,
            dtype=h.dtype,
        )

        if self.use_cnn:
            conv_input = h.transpose(1, 2)
            conv_features = [
                torch.relu(conv(conv_input)).transpose(1, 2)
                for conv in self.convs
            ]
            local_features = self.cnn_fusion(torch.cat(conv_features, dim=-1))
            h = self.cnn_norm(h + self.dropout(local_features))
            local_embedding = self._attention_pool(h)
        else:
            local_embedding = zero_embedding

        if self.use_lstm:
            temporal_features, _ = self.bilstm(h)
            temporal_features = self.lstm_projection(temporal_features)
            temporal_features = self.temporal_norm(h + self.dropout(temporal_features))
            temporal_embedding = self._attention_pool(temporal_features)
        else:
            temporal_features = h
            temporal_embedding = zero_embedding

        source_context = self.source_projection(source_emb)
        if self.use_gate:
            gate = self.gate(torch.cat([temporal_features, source_context], dim=-1))
            fused = temporal_features + gate * source_context
        else:
            fused = self.simple_fusion_norm(0.5 * (temporal_features + source_context))

        if self.use_transformer:
            transformer_input = self._add_temporal_position(fused)
            transformed = self.transformer(transformer_input)
            global_features = self.transformer_norm(
                transformer_input + self.dropout(transformed)
            )
            global_embedding = self._attention_pool(global_features)
        else:
            global_features = fused
            global_embedding = zero_embedding

        embedding, attention = self._stage_fusion(
            [local_embedding, temporal_embedding, global_embedding]
        )
        logits = self.classifier(embedding)
        return logits, embedding, attention

    def _attention_pool(self, features: torch.Tensor) -> torch.Tensor:
        attention_logits = self.pool_attention(features).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=1)
        attended = torch.sum(features * attention.unsqueeze(-1), dim=1)
        mean_pooled = torch.mean(features, dim=1)
        return self.embedding_norm(attended + mean_pooled)

    def _add_temporal_position(self, features: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[1]
        positions = torch.linspace(
            0.0,
            1.0,
            seq_len,
            device=features.device,
            dtype=features.dtype,
        ).view(1, seq_len, 1)
        return features + self.position_projection(positions)

    def _stage_fusion(self, embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        concatenated = torch.cat(embeddings, dim=-1)
        embedding = self.stage_fusion(concatenated)
        batch_size = embedding.shape[0]
        stage_weight = torch.full(
            (batch_size, len(embeddings)),
            1.0 / len(embeddings),
            device=embedding.device,
            dtype=embedding.dtype,
        )
        return self.embedding_norm(embedding), stage_weight
