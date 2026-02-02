import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupGenerator(nn.Module):
    def __init__(self, d_type='sti_attention', th=1., in_channels=16, hid_channels=32, n_head=1, dropout=0):
        super().__init__()
        self.d_type = d_type
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.n_head = n_head
        self.dropout = dropout

        # STI-Attention components
        if d_type == 'sti_attention':
            self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=n_head, dropout=dropout)
            self.fc = nn.Linear(in_channels, in_channels)
            self.norm = nn.LayerNorm(in_channels)

        if d_type in ['learned', 'sti_attention']:
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, 1),
                nn.ReLU(),
                nn.BatchNorm2d(hid_channels),
                nn.Dropout(dropout, inplace=True),
                nn.Conv2d(hid_channels, n_head, 1),
            )
        elif d_type == 'estimate_th':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, n_head, 1),)
        elif d_type == 'learned_l2norm':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 1), padding=(1, 0)))
        
        self.th = th if isinstance(th, float) else nn.Parameter(torch.Tensor([1]))

    def forward(self, v, v_abs, tau=0.1, hard=True):
        assert v.size(0) == 1
        n_ped = v.size(-1)

        if self.d_type == 'sti_attention':
            # STI-Attention mechanism
            v_abs_attn = v_abs.squeeze(0).permute(2, 0, 1)  # (n_ped, seq_len, features)
            attn_output, attn_weights = self.attention(v_abs_attn, v_abs_attn, v_abs_attn)
            attn_output = self.fc(attn_output)
            attn_output = self.norm(attn_output + v_abs_attn)
            attn_output = attn_output.permute(1, 2, 0).unsqueeze(0)  # (1, seq_len, features, n_ped)
            v_abs = attn_output

        # Measure similarity between pedestrian pairs
        if self.d_type == 'euclidean':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned_l2norm':
            temp = self.group_cnn(v_abs).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type in ['learned', 'sti_attention']:
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(temp.size(0), -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)  # symmetric
        elif self.d_type == 'estimate_th':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-2, -1))
            dist_mat = temp.norm(p=2, dim=1)
            self.th = self.group_cnn(temp.reshape(temp.size(0), -1, n_ped, n_ped)).mean().exp()
        else:
            raise NotImplementedError

        dist_mat = dist_mat.squeeze(dim=0).mean(dim=0)
        indices = self.find_group_indices(v, dist_mat)
        v = self.group_backprop_trick_threshold(v, dist_mat, tau=tau, hard=hard)
        return v, indices

    def find_group_indices(self, v, dist_mat):
        n_ped = v.size(-1)
        mask = torch.ones_like(dist_mat).mul(1e4).triu()
        top_row, top_column = torch.nonzero(dist_mat.tril(diagonal=-1).add(mask).le(self.th), as_tuple=True)
        indices_raw = torch.arange(n_ped, dtype=top_row.dtype, device=v.device)
        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c
        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_row.dtype, device=v.device)
        indices = torch.zeros_like(indices_raw)
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j
        return indices

    def group_backprop_trick_threshold(self, v, dist_mat, tau=1, hard=False):
        sig = (-(dist_mat - self.th) / tau).sigmoid()
        sig_norm = sig / sig.sum(dim=0, keepdim=True)
        v_soft = v @ sig_norm
        return (v - v_soft).detach() + v_soft if hard else v_soft

    @staticmethod
    def ped_group_pool(v, indices):
        assert v.size(-1) == indices.size(0)
        n_ped = v.size(-1)
        n_ped_pool = indices.unique().size(0)
        v_pool = torch.zeros(v.shape[:-1] + (n_ped_pool,), device=v.device)
        v_pool.index_add_(-1, indices, v)
        v_pool_num = torch.zeros((v.size(0), 1, 1, n_ped_pool), device=v.device)
        v_pool_num.index_add_(-1, indices, torch.ones((v.size(0), 1, 1, n_ped), device=v.device))
        v_pool /= v_pool_num
        return v_pool

    @staticmethod
    def ped_group_unpool(v, indices):
        assert v.size(-1) == indices.unique().size(0)
        return torch.index_select(input=v, dim=-1, index=indices)

    @staticmethod
    def ped_group_mask(indices):
        mask = torch.eye(indices.size(0), dtype=torch.bool, device=indices.device)
        for i in indices.unique():
            idx_list = torch.nonzero(indices.eq(i))
            for idx in idx_list:
                mask[idx, idx_list] = 1
        return mask

class GroupIntegrator(nn.Module):
    def __init__(self, mix_type='mean', n_mix=3, out_channels=5, pred_seq_len=12):
        super().__init__()
        self.mix_type = mix_type
        self.pred_seq_len = pred_seq_len
        if mix_type == 'mlp':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * pred_seq_len * n_mix, out_channels * pred_seq_len,
                                                       kernel_size=1), )
        elif mix_type == 'cnn':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * n_mix, out_channels,
                                                       kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, v_stack):
        n_batch, n_ped = v_stack[0].shape[0], v_stack[0].shape[3]
        if self.mix_type == 'sum':
            v = torch.stack(v_stack, dim=0).sum(dim=0)
        elif self.mix_type == 'mean':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
        elif self.mix_type == 'mlp':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v_stack = torch.cat(v_stack, dim=1).reshape(n_batch, -1, 1, n_ped)
            v = v + self.st_gcns_mix(v_stack).view(n_batch, -1, self.pred_seq_len, n_ped)
        elif self.mix_type == 'cnn':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v = v + self.st_gcns_mix(torch.cat(v_stack, dim=1))
        else:
            raise NotImplementedError
        return v

def generate_identity_matrix(v):
    return [torch.eye(v.size(3), device=v.device).repeat(v.size(2), 1, 1),
            torch.eye(v.size(2), device=v.device).repeat(v.size(3), 1, 1)]

class GPGraph(nn.Module):
    def __init__(self, baseline_model, in_channels=16, out_channels=5, obs_seq_len=8, pred_seq_len=12,
                 d_type='sti_attention', d_th='learned', mix_type='mlp', group_type=None, weight_share=True):
        super().__init__()

        self.baseline_model = baseline_model
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
        self.mix_type = mix_type
        self.weight_share = weight_share

        group_type = (True,) * 3 if group_type is None else group_type
        self.include_original = group_type[0]
        self.include_inter_group = group_type[1]
        self.include_intra_group = group_type[2]

        self.group_gen = GroupGenerator(d_type=d_type, th=d_th, in_channels=in_channels, hid_channels=8)
        self.group_mix = GroupIntegrator(mix_type=mix_type, n_mix=sum(group_type),
                                         out_channels=out_channels, pred_seq_len=pred_seq_len)

    def forward(self, v_abs, v_rel):
        v_stack = []

        # Pedestrian graph
        if self.include_original:
            v = v_rel
            i = generate_identity_matrix(v)
            v = v.permute(0, 2, 3, 1)
            v = self.baseline_model(v, i) if self.weight_share else self.baseline_model[0](v, i)
            v = v.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v)

        # STI-Attention enhanced grouping
        v_rel, indices = self.group_gen(v_rel, v_abs, hard=True)

        if self.include_inter_group:
            v_e = self.group_gen.ped_group_pool(v_rel, indices)
            i_e = generate_identity_matrix(v_e)
            v_e = v_e.permute(0, 2, 3, 1)
            v_e = self.baseline_model(v_e, i_e) if self.weight_share else self.baseline_model[1](v_e, i_e)
            v_e = v_e.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_e = self.group_gen.ped_group_unpool(v_e, indices)
            v_stack.append(v_e)

        if self.include_intra_group:
            v_i = v_rel
            mask = self.group_gen.ped_group_mask(indices)
            i_i = generate_identity_matrix(v_i)
            v_i = v_i.permute(0, 2, 3, 1)
            v_i = self.baseline_model(v_i, i_i, mask) if self.weight_share else self.baseline_model[2](v_i, i_i, mask)
            v_i = v_i.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v_i)

        # Group Integration
        v = self.group_mix(v_stack)

        return v, indices
