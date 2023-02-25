import torch
from torch import nn
import torch.nn.functional as F
from module.Attention import NeighborEncoder, ViewAttention


class MVEncoder(nn.Module):
    def __init__(self, feature_dim, embedding_size, embedding_u_size, num_view, feat_drop, attn_size, attn_drop):
        """
        feature_dim: 节点的特征维度
        embedding_size: baseEmbedding嵌入的维度
        embedding_u_size: edgeEmbedding嵌入的维度
        num_view: 视图个数
        """
        super(MVEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.num_view = num_view
        self.attn_size = attn_size
        self.feature_dim = feature_dim

        self.embed_trans = nn.Parameter(torch.FloatTensor(self.feature_dim, embedding_size), requires_grad=True)
        nn.init.xavier_normal_(self.embed_trans, gain=1.414)
        self.u_embed_trans = nn.Parameter(torch.FloatTensor(self.num_view, self.feature_dim, embedding_u_size), requires_grad=True)
        nn.init.xavier_normal_(self.u_embed_trans, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.NeighborEncoder = NeighborEncoder(self.embedding_u_size, self.attn_size, attn_drop)
        self.View_attention = ViewAttention(self.embedding_u_size, self.attn_size, attn_drop)

        self.trans_weights = nn.Parameter(
            torch.FloatTensor(self.embedding_u_size, self.embedding_size)
        )
        nn.init.xavier_normal_(self.trans_weights, gain=1.414)
        self.fc = nn.Linear(self.embedding_size*2, self.embedding_size)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs, node_neigh):
        # inputs.size=(batch_size, feature_dim)
        # node_neigh.size=(batch_size, num_view, num_neigh, feature_dim)
        node_embed = torch.mm(inputs, self.embed_trans)  # (batch_size, embedding_size)
        # [num_view, batch_size * num_neigh, feature_dim]*[num_view, feature_dim, embedding_u_size]->[batch_size, num_view, num_neigh, embedding_u_size]
        node_embed_neighbors = torch.bmm(node_neigh.permute(1, 0, 2, 3).reshape(self.num_view, -1, self.feature_dim),
                                         self.u_embed_trans).reshape(self.num_view, inputs.shape[0], -1,
                                                                     self.embedding_u_size).permute(1, 0, 2, 3)
        node_view_embed = self.NeighborEncoder(self.feat_drop(node_embed_neighbors))  # (batch_size, num_view, embedding_u_size)
        node_u_embed = self.View_attention(node_view_embed)  # (batch_size, embedding_u_size)
        node_embed = self.fc(torch.concat([node_embed, torch.matmul(node_u_embed, self.trans_weights)], dim=1))
        last_node_embed = F.normalize(node_embed, dim=1)
        return last_node_embed
