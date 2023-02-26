import torch
from torch import nn
import torch.nn.functional as F
from module.MultiViewEncoder import MVEncoder


class MoGCL(nn.Module):
    def __init__(self, features, dim=128, num_view=3, num_pos=5, num_neigh=10, attn_size=64, feat_drop=0.3,
                 attn_drop=0.3, K=65536, m=0.999, T=0.5, mlp=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: MoGCL momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoGCL, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.register_buffer("features", torch.FloatTensor(features))
        self.num_view = num_view
        self.num_pos = num_pos
        self.num_neigh = num_neigh
        self.dim = dim
        self.encoder_q = MVEncoder(features.shape[-1], dim, dim, num_view, feat_drop, attn_size, attn_drop)
        self.encoder_k = MVEncoder(features.shape[-1], dim, dim, num_view, feat_drop, attn_size, attn_drop)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ELU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ELU(), self.encoder_k.fc
            )
            for model in self.encoder_q.fc:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain=1.414)
                    if model.bias is not None:
                        model.bias.data.fill_(0.0)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, indexes, keys):
        self.queue[:, indexes] = keys.T

    def forward(self, node_inputs, pos_node_inputs, neg_node_inputs):
        # node_inputs: (nodes, nodes_neigh).shape = ((batch_size),(batch_size, num_view, num_neigh))
        # pos_node_inputs: (pos_nodes, pos_nodes_neigh).shape = ((batch_size, num_pos), (batch_size, num_pos, num_view, num_neigh))
        q = self.encoder_q(self.features[node_inputs[0]], self.features[node_inputs[1]])

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            pos_nodes, pos_nodes_neigh = pos_node_inputs
            indexed = pos_nodes.reshape(-1)
            k = self.encoder_k(self.features[indexed],
                               self.features[pos_nodes_neigh.reshape(-1, self.num_view, self.num_neigh)])

        self._dequeue_and_enqueue(indexed, k)
        k = k.reshape(-1, self.num_pos, self.dim)  # (batch_size, num_pos, dim)
        # positive logits: N x num_pos
        l_pos = torch.einsum("nc,npc->np", [q, k])
        # negative logits: N x (num_nodes-num_pos)
        l_neg = torch.einsum("nc,cnk->nk", [q, self.queue[:, neg_node_inputs].clone().detach()])
        # logits: N x num_nodes
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        return logits

    @torch.no_grad()
    def get_embeds(self, nodes, nodes_neigh):
        return self.encoder_q(self.features[nodes], self.features[nodes_neigh])
