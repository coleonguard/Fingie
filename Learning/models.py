import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


# Temporal Convolutional Network (TCN) Model
class Chomp1d(nn.Module):
    """Chomps off the extra padding in the TCN layers."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Remove the extra padding added to achieve causal convolution
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single temporal block in the TCN."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First convolution layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers into a sequential module
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Forward pass through the temporal block
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """TCN model for sequence modeling."""

    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        # Create multiple temporal blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)  # Transpose to (batch_size, input_size, seq_length)
        y = self.network(x)
        y = y.transpose(1, 2)  # Back to (batch_size, seq_length, num_channels[-1])
        y = self.linear(y)
        return y


# Transformer Model
class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input embeddings."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle case when d_model is odd
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embedding_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for sequence modeling."""

    def __init__(self, input_size, output_size, num_layers, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        encoder_layers = nn.TransformerEncoderLayer(dim_feedforward, nhead,
                                                    dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(dim_feedforward, output_size)

    def forward(self, src):
        # src shape: (batch_size, seq_length, input_size)
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Transformer expects (seq_length, batch_size, embedding_dim)
        output = self.transformer_encoder(src)
        output = output.transpose(0, 1)  # Back to (batch_size, seq_length, embedding_dim)
        output = self.decoder(output)
        return output

# Spatial-Temporal Graph Convolutional Network (ST-GCN)
class STGCNBlock(nn.Module):
    """A single block of the ST-GCN model."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(STGCNBlock, self).__init__()
        self.gcn = pyg_nn.GCNConv(in_channels, out_channels)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                             stride=(stride, 1), padding=((kernel_size - 1) // 2, 0))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_nodes, sequence_length, in_channels)
        N, T, C = x.size(1), x.size(2), x.size(3)
        x = x.view(-1, C)  # Reshape for GCN: (batch_size * num_nodes * sequence_length, in_channels)
        x = self.gcn(x, edge_index)  # Apply GCN
        x = x.view(-1, N, T, x.size(-1))  # Reshape back
        x = x.permute(0, 3, 2, 1)  # (batch_size, out_channels, sequence_length, num_nodes)
        x = self.tcn(x)  # Apply TCN
        x = x.permute(0, 3, 2, 1)  # Back to (batch_size, num_nodes, sequence_length, out_channels)
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """ST-GCN model for spatial-temporal modeling."""

    def __init__(self, num_nodes, in_channels, out_channels, kernel_size=3, num_class=5):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels, 64, kernel_size)
        self.block2 = STGCNBlock(64, 128, kernel_size)
        self.block3 = STGCNBlock(128, 256, kernel_size)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_nodes, sequence_length, in_channels)
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        # Global average pooling over time and nodes
        x = x.mean(dim=2)  # Average over time
        x = x.mean(dim=1)  # Average over nodes
        x = self.fc(x)
        return x


# Recurrent Neural Network with Attention Mechanism
class RNNWithAttention(nn.Module):
    """RNN model with attention for sequence modeling."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNWithAttention, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_length, hidden_size)
        attn_weights = torch.softmax(self.attention(rnn_out), dim=1)  # (batch_size, seq_length, 1)
        context = torch.sum(attn_weights * rnn_out, dim=1)  # (batch_size, hidden_size)
        output = self.fc(context)  # (batch_size, output_size)
        return output.unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat over sequence length
    
def initialize_model(model_type, input_dim, output_dim, cfg):
    """
    Initializes the model based on the model_type and configuration parameters.

    Args:
        model_type (str): Type of the model ('TCN', 'Transformer', 'STGCN', 'RNNWithAttention').
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
        cfg (dict): Configuration dictionary containing model hyperparameters.

    Returns:
        nn.Module: Initialized model.
    """
    if model_type == 'TCN':
        num_channels = cfg.get('num_channels', [64, 64, 64])
        kernel_size = cfg.get('kernel_size', 2)
        dropout = cfg.get('dropout', 0.2)
        model = TCN(input_size=input_dim, output_size=output_dim, num_channels=num_channels,
                    kernel_size=kernel_size, dropout=dropout)
    elif model_type == 'Transformer':
        num_layers = cfg.get('num_layers', 3)
        nhead = cfg.get('nhead', 4)
        dim_feedforward = cfg.get('dim_feedforward', 128)
        dropout = cfg.get('dropout', 0.1)
        model = TransformerModel(input_size=input_dim, output_size=output_dim, num_layers=num_layers,
                                 nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    elif model_type == 'STGCN':
        num_nodes = cfg.get('num_nodes', input_dim)  # Assuming each feature is a node
        kernel_size = cfg.get('kernel_size', 3)
        model = STGCN(num_nodes=num_nodes, in_channels=1, out_channels=64,
                      kernel_size=kernel_size, num_class=output_dim)
    elif model_type == 'RNNWithAttention':
        hidden_size = cfg.get('hidden_size', 128)
        num_layers = cfg.get('num_layers', 2)
        model = RNNWithAttention(input_size=input_dim, hidden_size=hidden_size,
                                 num_layers=num_layers, output_size=output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model