"""Mechanism controls initialized identically to the comparison model."""
import torch
from .model import MultiSourceRiskNet


class ComparisonAblationNet(MultiSourceRiskNet):
    def __init__(self, *args, variant="Full", **kwargs):
        for name in ["use_cnn", "use_lstm", "use_transformer", "use_gate"]:
            kwargs.pop(name, None)
        super().__init__(*args, **kwargs)
        self.variant = variant

    def forward(self, x, source_ids):
        v = self.variant
        if v in ["Full", "H", "I", "J", "K"]:
            return super().forward(x, source_ids)
        source_emb = self.source_embedding(source_ids)
        h = self.input_projection(torch.cat([x, source_emb], dim=-1))
        zero = torch.zeros_like(h[:, 0])
        if v not in ["A", "D", "F", "G"]:
            conv = [torch.relu(c(h.transpose(1, 2))).transpose(1, 2) for c in self.convs]
            update = self.dropout(self.cnn_fusion(torch.cat(conv, dim=-1)))
            h = self.cnn_norm(update if v == "L" else h + update)
            local = self._attention_pool(h)
        else:
            local = zero
        if v not in ["B", "D", "E", "G"]:
            temporal, _ = self.bilstm(h)
            temporal = self.temporal_norm(h + self.dropout(self.lstm_projection(temporal)))
        else:
            temporal = h
        context = self.source_projection(source_emb)
        if v == "N":
            context = torch.zeros_like(context)
        gate = self.gate(torch.cat([temporal, context], dim=-1))
        if v == "M":
            gate = torch.ones_like(gate)
        fused = self.temporal_norm(temporal + gate * context)
        temporal_embedding = self._attention_pool(fused) if v not in ["B", "D", "E", "G"] else zero
        if v not in ["C", "E", "F", "G"]:
            positioned = self._add_temporal_position(fused)
            global_features = self.transformer_norm(positioned + self.dropout(self.transformer(positioned)))
            global_embedding = self._attention_pool(global_features)
        else:
            global_embedding = zero
        if v == "G":
            # Keep a data-dependent shallow readout, not the legacy all-zero classifier.
            global_embedding = self._attention_pool(h)
        embedding, attention = self._stage_fusion([local, temporal_embedding, global_embedding])
        return self.classifier(embedding), embedding, attention
