import torch
import torch.nn as nn
    
class CellType:
    RNN = "rnn"
    GRU = "gru"
    LSTM = "lstm"
    
    @staticmethod
    def rnn_layer(cell_type):
        cell_types = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        return cell_types[cell_type]
    

class RecurrentEncoder(nn.Module):
    def __init__(self, cell_type, embedding_dim, hidden_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, device='cpu'):
        super(RecurrentEncoder, self).__init__()
        self.device = device
        self.cell_type = cell_type
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional

        ### Layers ###
        self.rnn = CellType.rnn_layer(cell_type)(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x, hidden):
        output, hidden = self.rnn(x.unsqueeze(1), hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)