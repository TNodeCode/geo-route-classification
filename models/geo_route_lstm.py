import torch
import torch.nn as nn

from models.embedding import GeoRouteEmbedding
from models.rnn import RecurrentEncoder, CellType
from models.classifier import Classifier


class GeoRouteLSTM(nn.Module):
    def __init__(self, device="cpu", num_layers=3, embedding_dim=32, hidden_size=256, max_length=39, bidirectional=True, cell_type = CellType.LSTM, dropout=0.1):
        super(GeoRouteLSTM, self).__init__()
        self.device = device

        ### Model hyperparameters
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.dropout = dropout

        ### Layers
        self.embedding = GeoRouteEmbedding()
        self.encoder = RecurrentEncoder(
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device
        )
        _hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = Classifier(2, embedding_dim=_hidden_size+28, hidden_size=512).to(device)


    def forward(self, lat, long, asn, ip_source, geo_cc, src_as, dest_as, src_cc, dest_cc):
        # get embeddings of sequential and non-sequential features
        input_seqs = self.embedding.get_sequence_embedding(x_lat=lat, x_long=long, x_asn=asn, x_ip_source=ip_source, x_geo_cc=geo_cc)        
        non_seq_features = self.embedding.get_non_sequential_embedding(x_src_as=src_as, x_dest_as=dest_as, x_src_cc=src_cc, x_dest_cc=dest_cc)
        
        # Initialize the encoder hidden state and cell state with zeros
        hn = self.encoder.initHidden(input_seqs.shape[0], device=self.device)
        cn = self.encoder.initHidden(input_seqs.shape[0], device=self.device)
        hidden = (hn, cn) if self.cell_type == CellType.LSTM else hn

        # Iterate over the sequence positios and run every position through the encoder
        for i in range(input_seqs.shape[1]):
            # Run the i-th position of the input sequence through the encoder.
            # As a result we will get the prediction (output), the hidden state (hn).
            # The hidden state and cell state will be used as inputs in the next round
            output, hidden = self.encoder(
                x=input_seqs[:, i],
                hidden=hidden
            )

        # Run output of encoder and embedding of non-sequential features through classifier
        logits = self.classifier(torch.cat([output, non_seq_features], axis=2))
        return logits
