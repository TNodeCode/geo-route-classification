import torch
import torch.nn as nn
from models.classifier import Classifier


class GeoRouteTransformer(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embedding_dim=256,
        num_layers=6,
        heads=8,
        dropout=0,
        device="cuda",
    ):
        super(GeoRouteTransformer, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(40, 32)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            batch_first=True
        )
        self.target_embedding = nn.Embedding(40, 32)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.cls = Classifier(
            trg_vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=1024,
            softmax_dim=1
        )

    def forward(self, sequences, non_seq_features, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        non_seq_features = torch.cat([
            non_seq_features,
            torch.zeros(sequences.size(0), 1, 4).to(self.device)
        ], axis=2)
        pos = self.pos_embedding(
            torch.arange(40).to(self.device)
        ).unsqueeze(0).repeat(sequences.shape[0], 1, 1)
        target_seq = self.pos_embedding(
            torch.arange(40).to(self.device)
        ).unsqueeze(0).repeat(sequences.shape[0], 1, 1)
        out = self.transformer(
            torch.cat([sequences, non_seq_features], axis=1) + pos,
            target_seq,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        # Run encoded embeddings through classifier
        out = self.cls(out[:, 0, :])
        return out 