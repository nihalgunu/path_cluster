import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvolutionNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim_hidden,
        nconv=2,
        obs_seq_len=8,
        pred_seq_len=12,
        kernel_size=3,
        stride=1,
        dropout=0.1,
        activation="relu",
        ):
        super(TemporalConvolutionNet, self).__init__()
        assert kernel_size % 2 == 1
        assert nconv >= 2
        padding = ((kernel_size - 1) // 2, 0)
        
        self.norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(nconv)])
        self.timeconvs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1), padding) for _ in range(nconv)
        ])

        # Adding an additional convolutional layer for spatial feature enhancement
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)  # Enhances spatial features

        self.timelinear1 = nn.Linear(obs_seq_len, pred_seq_len)
        self.timelinear2 = nn.Linear(pred_seq_len, pred_seq_len)
        self.timedropout1 = nn.Dropout(dropout)
        self.timedropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(in_channels, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        # Adding a self-attention mechanism to better capture temporal dependencies
        self.self_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4)  # Adds self-attention for temporal sequences

        # Adding a layer normalization for stability before self-attention
        self.layer_norm_before_attention = nn.LayerNorm(in_channels)  # Normalization before attention

    def forward(self, x):
        for i in range(len(self.norms)):
            x_norm = self.norms[i](x)
            x_perm = x_norm.permute(0, 3, 1, 2)
            x_perm = _get_activation_fn(self.activation)(self.timeconvs[i](x_perm))
            x_perm = x_perm.permute(0, 2, 3, 1)
            x = x + x_perm
        
        # Apply spatial convolution for enhanced feature extraction
        x = x.permute(0, 3, 1, 2)  # Adjust for spatial conv
        x = self.spatial_conv(x)  # Apply spatial conv
        x = x.permute(0, 2, 3, 1)  # Revert permutation

        # Normalize and apply self-attention
        x = x.permute(1, 0, 2, 3)  # Adjust for self-attention (seq_len, batch, channels, nodes)
        x = self.layer_norm_before_attention(x)  # Apply layer normalization
        x, _ = self.self_attention(x, x, x)  # Apply self-attention
        x = x.permute(1, 0, 2, 3)  # Revert to original order (batch, seq_len, channels, nodes)
        
        x = self.timedropout1(_get_activation_fn(self.activation)(self.timelinear1(x)))
        x = self.timedropout2(_get_activation_fn(self.activation)(self.timelinear2(x)))
        x = x.permute(0, 3, 1, 2)
        x = self.dropout(_get_activation_fn(self.activation)(self.linear1(x)))
        out = self.linear2(x)
        return out

