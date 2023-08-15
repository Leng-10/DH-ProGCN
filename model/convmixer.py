import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch import Tensor


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        y = self.fn(x)
        return y + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=4, n_classes=2):

    return nn.Sequential(
        nn.Conv3d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm3d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    # nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.Conv3d(dim, dim, kernel_size, groups=dim, padding=4),
                    # nn.Conv3d(dim, dim, kernel_size, groups=dim),
                    nn.GELU(),
                    nn.BatchNorm3d(dim)
                )),
                nn.Conv3d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


def ConvMix(dim1,dim2, depth, kernel_size, patch_size, n_classes=2):

    return nn.Sequential(
        *[nn.Sequential(

                Residual(nn.Sequential(
                    # nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.Conv3d(dim1, dim2, kernel_size, groups=dim2, padding="same"),
                    # nn.Conv3d(dim, dim, kernel_size, groups=dim),
                    nn.GELU(),
                    nn.BatchNorm3d(dim2)
                )),
                nn.Conv3d(dim2, dim2, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(dim2)
        ) for i in range(depth)]
    )




class ECA(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)
        return src1


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
