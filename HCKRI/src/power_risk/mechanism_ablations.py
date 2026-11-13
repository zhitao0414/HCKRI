"""Mechanism controls preserving the historical A-G Full architecture."""
import torch

from .ablations import AblationRiskNet


class MechanismRiskNet(AblationRiskNet):
    def __init__(self, *args, mechanism="Full", **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = mechanism

    def forward(self, x, source_ids):
        source_emb = self.source_embedding(source_ids)
        h = self.input_projection(torch.cat([x, source_emb], dim=-1))
        conv_features = [torch.relu(conv(h.transpose(1, 2))).transpose(1, 2)
                         for conv in self.convs]
        local_features = self.cnn_fusion(torch.cat(conv_features, dim=-1))
        local_update = self.dropout(local_features)
        h = self.cnn_norm(local_update if self.mechanism == "No-CNN-Residual"
                          else h + local_update)
        local_embedding = self._attention_pool(h)

        temporal_features, _ = self.bilstm(h)
        temporal_features = self.lstm_projection(temporal_features)
        temporal_features = self.temporal_norm(h + self.dropout(temporal_features))
        # Match the A-G runner: its temporal pooling occurs before source gating.
        temporal_embedding = self._attention_pool(temporal_features)
        source_context = self.source_projection(source_emb)
        if self.mechanism == "No-Source-Context":
            source_context = torch.zeros_like(source_context)
        gate = self.gate(torch.cat([temporal_features, source_context], dim=-1))
        if self.mechanism == "Fixed-Source-Gate":
            gate = torch.ones_like(gate)
        fused = temporal_features + gate * source_context

        transformer_input = self._add_temporal_position(fused)
        transformed = self.transformer(transformer_input)
        global_features = self.transformer_norm(transformer_input + self.dropout(transformed))
        global_embedding = self._attention_pool(global_features)
        if self.mechanism == "Final-Stage-Only":
            local_embedding = torch.zeros_like(local_embedding)
            temporal_embedding = torch.zeros_like(temporal_embedding)
        embedding, attention = self._stage_fusion(
            [local_embedding, temporal_embedding, global_embedding])
        return self.classifier(embedding), embedding, attention
