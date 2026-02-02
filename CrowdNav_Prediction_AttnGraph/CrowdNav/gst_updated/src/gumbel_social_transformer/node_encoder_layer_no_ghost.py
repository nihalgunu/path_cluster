import torch
import torch.nn as nn
from gst_updated.src.gumbel_social_transformer.mha import VanillaMultiheadAttention
from gst_updated.src.gumbel_social_transformer.utils import _get_activation_fn

class NodeEncoderLayer(nn.Module):
    r"""No ghost version"""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu", attn_mech='vanilla'):
        super(NodeEncoderLayer, self).__init__()
        self.attn_mech = attn_mech
        if self.attn_mech == 'vanilla':
            self.self_attn = VanillaMultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm_node = nn.LayerNorm(d_model)
        else:
            raise RuntimeError('NodeEncoderLayer currently only supports vanilla mode.')
        self.norm1_node = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)  # Base dropout rate
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.nhead = nhead
    
    def forward(self, x, sampled_edges, attn_mask, device="cuda:0"):
        """
        Encode pedestrian edge with node information.
        """
        if self.attn_mech == 'vanilla':
            bsz, nnode, _ = x.shape
            attn_mask_ped = (attn_mask.sum(-1) > 0).float().unsqueeze(-1).to(device)
            x = self.norm_node(x)
            x = x * attn_mask_ped
            x_perm = x.permute(1, 0, 2)

            # Calculate crowd density
            crowd_density = sampled_edges.sum() / (bsz * x.shape[1])
            # Adjust dropout rate dynamically based on crowd density
            dynamic_dropout_rate = self.dropout.p * (1 + crowd_density / x.shape[1])
            dynamic_dropout_rate = min(dynamic_dropout_rate, 0.5)  # Cap the dropout rate for stability

            adj_mat = sampled_edges.sum(2)
            adj_mat = torch.stack([adj_mat for _ in range(self.nhead)], dim=1)
            adj_mat = adj_mat.view(bsz*self.nhead, adj_mat.shape[2], adj_mat.shape[3])
            
            # Apply self-attention
            x2, attn_weights, _ = self.self_attn(x_perm, x_perm, x_perm, attn_mask=adj_mat)
            x2 = x2.permute(1, 0, 2)
            # Apply dynamic dropout to the output of self-attention
            x = x + nn.functional.dropout(x2, p=dynamic_dropout_rate, training=self.training)
            
            x2 = self.norm1_node(x)
            # Apply dynamic dropout after the first linear layer and activation
            x2 = nn.functional.dropout(self.activation(self.linear1(x2)), p=dynamic_dropout_rate, training=self.training)
            # Apply dynamic dropout after the second linear layer
            x2 = nn.functional.dropout(self.linear2(x2), p=dynamic_dropout_rate, training=self.training)
            x = x + x2
            
            return x, attn_weights
        else:
            raise RuntimeError('NodeEncoderLayer currently only supports vanilla mode.')

# End of NodeEncoderLayer class

