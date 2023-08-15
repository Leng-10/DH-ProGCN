import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        # self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class Attention(nn.Module):
    def __init__(self, emb_size):
        super(Attention, self).__init__()
        self.weight_W = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
    def forward(self, inputs, masks=None):
        batch_size = inputs.size(0)
        scores = torch.m



class dgcn_cls(nn.Module):

    def __init__(self, part_num):
        super(dgcn_cls, self).__init__()
        self.part_num = part_num

        self.global_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=(0, 1))

        # self.fc1 = nn.Linear(196, 64)
        # self.fc2 = nn.Linear(196, 64)
        # self.fc3 = nn.Linear(196, self.part_num)
        # self.fc_cls = nn.Linear(self.part_num + 1, 2)
        # self.conv1 = DenseGCNConv(196, 196)
        # self.conv2 = DenseGCNConv(196, 196)

        self.fc1 = nn.Linear(891, 128)
        self.fc2 = nn.Linear(891, 128)
        self.fc3 = nn.Linear(891, self.part_num)

        self.fc_cls = nn.Linear(self.part_num + 1, 2)

        self.conv1 = DenseGCNConv(891, 891)
        self.conv2 = DenseGCNConv(891, 891)
        self.mha = MultiHeadedAttention(1, self.part_num)

    def forward(self, feature_map, weighted_feature):
        # the inputs are the feature outputed by layer4 of resnet101 and the attention_mask outputed by the channel_grouping_layer
        # the dimension is 2048 * 14 *14 and part_num * 14 * 14, respectively

        # region_features = weighted_feature.reshape(-1, self.part_num, 196)
        region_features = weighted_feature.reshape(-1, self.part_num, 891)

        # get the dynamic correlation matrix of the graph for the image using self-attention
        q = self.fc1(region_features)
        k = self.fc2(region_features)
        v = self.fc3(region_features)
        corr_matrix = self.mha(q, k, v)

        # propogate the features by GCN
        node_feature = torch.relu(self.conv1(region_features, corr_matrix))
        node_feature = nn.Dropout()(node_feature)
        node_feature = torch.relu(self.conv2(node_feature, corr_matrix))
        node_feature = nn.Dropout()(node_feature)

        # node_feature = node_feature.reshape(-1, self.part_num, 14, 14)
        node_feature = node_feature.reshape(-1, self.part_num, 9, 11, 9)

        # concatenate global feature and the integrated local features for classification

        # feature_map = feature_map.reshape(-1, 2048, 1, 196)
        feature_map = feature_map.reshape(-1, 512, 1, 891)
        feature_map = feature_map.transpose(1, 3)
        compressed_feature = self.global_avgpool2d(feature_map)
        # compressed_feature = compressed_feature.reshape(-1, 1, 14, 14)
        compressed_feature = compressed_feature.reshape(-1, 1, 9, 11, 9)

        final_feature = torch.cat((compressed_feature, node_feature), dim=1)

        res = self.global_avgpool3d(final_feature)
        res = torch.flatten(res, 1)
        res = self.fc_cls(res)

        return res