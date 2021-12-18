import torch
import torch.nn as nn
from torch.distributions import Categorical
from math import sqrt
from hyperparams import *


class NodeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(NodeEmbedding, self).__init__()
        self.embedding = nn.Linear(2, embed_dim)
    def forward(self, inputs):
        """
        Args:
            inputs : num_items x 2
        """
        return self.embedding(inputs)


class Attention(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, num_heads):
        super(Attention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.first_normalization = nn.LayerNorm([args.num_items, args.embed_dim])
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.second_normalization = nn.LayerNorm([args.num_items, args.embed_dim])
    def forward(self, inputs):
        """
        Args:
            inputs: Output of NodeEmbedding : [batch_size x num_items x embed_dim]

        ** num_items : TOTAL number of items:
            eg : 3 dogs, 5 books, 2 notebooks --> n_items = 3+5+2 = 10
        """
        # pytorch MultiheadAttention takes input [seq_len x batch_size x embedding_dim]
        # So we need to change it to our models.
        x = inputs.permute(1, 0, 2)     #[num_items x batch_size x embed_dim]
        y, _ = self.mha(x, x, x)             #[num_items x batch_size x embed_dim]
        y = y + x
        y = y.permute(1, 0, 2)          #[batch_size x num_items x embed_dim]
        z = self.first_normalization(y)     #[batch_size x num_items x embed_dim]
        z = self.ffnn(z)                    #[batch_size x num_items x embed_dim]
        z = y + z                           #[batch_size x num_items x embed_dim]
        z = self.second_normalization(z)    #[batch_size x num_items x embed_dim]
        return z

class Encoder(nn.Sequential):
    def __init__(self, embed_dim, num_heads, feedforward_dim =  512, n_encoder_layer = 2):
        layers = []
        for _ in range(n_encoder_layer):
            layers.append(Attention(embed_dim, feedforward_dim, num_heads))
        super(Encoder, self).__init__(*layers)


class Glimpse(nn.Module):
    """
    For simplicity, we compute h_c^(N) just by Graphembedding, h^(bar)(N)
    """
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super(Glimpse, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim / num_heads
        #h_c^(N) : [batch_size x embed_dim]
        self.W_q = nn.Linear(embed_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.W_v = nn.Linear(embed_dim, hidden_dim)

        self.W_out = nn.Linear(embed_dim, embed_dim)
    def forward(self, context, node_embedding, mask=None):
        """
        Args:
            context = Graph embedding = [batch_size x embed_dim]
            node_embedding = Node embedding = [batch_size x num_items x embed_dim]
            mask = [batch_size x num_items], BoolTensor

        ** num_items : TOTAL number of items:
            eg : 3 dogs, 5 books, 2 notebooks --> n_items = 3+5+2 = 10

        """
        batch_size, num_items, embed_dim = node_embedding.size()
        query = self.W_q(context)       #[batch_size x hidden_dim]
        key = self.W_k(node_embedding)  #[batch_size x num_items x hidden_dim]
        value = self.W_v(node_embedding)    #[batch_size x num_items x hidden_dim]

        query = query.view(
            batch_size, self.num_heads, self.d_k
        )       #[batch_size x n_head x d_k]

        key = key.view(
            batch_size, num_items, self.num_heads, self.d_k
        ).permute(0, 2, 1, 3).contiguous()      #[batch_size x n_head x n_items x d_k]

        value = value.view(
            batch_size, num_items, self.num_heads, self.d_k
        ).permute(0, 2, 1, 3).contiguous()      #[batch_size x n_head x n_items x d_k]

        # Note : By defining W_q, W_k, W_v has image in R^(hidden_dim),
        # we avoid the concatinate.
        # 그러니깐 hidden_dim으로 보낸 다음에 num_heads x d_k 로 쪼갰다는 말.
        # 안그러면 d_k로 보내는 행렬을 num_heads만큼 직접 만들어줘야해서 불편하다.

        qk = torch.einsum(
            "ijl, ijkl -> ijk", [query, key]
        ) * (1 / sqrt(self.d_k))        #[batch_size x n_head x n_items]

        if mask is not None:        # Apply the mask to the qk.
            _mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            qk[_mask] = -1000000.0                              # PLEASE CHECK THE SIZE


        att_weight = torch.softmax(qk, -1)
        ret = torch.einsum(
            "ijk, ijkl -> ijl", att_weight, value
        )   # [batch x n_head x d_k]

        ret = ret.view(batch_size, -1)  # [batch x embed_dim]
        # ret is h_c^(N+1)
        return ret

class Pointer(nn.Module):
    """
    Calculate the log probabilities.
    """
    def __init__(self, embed_dim, hidden_dim, C = 10):
        super(Pointer, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(embed_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.W_v = nn.Linear(embed_dim, hidden_dim)
        self.clip_constant = C

    def forward(self, context, node_embedding, mask=None):
        """
        Remark that here we have only one attention head.
        Args:
            ** context : Not h_c^(N). It is h_c^(N+1) which is calculated in the Glimpse.
                         [batch x embedding_dim]
            node_embedding = [batch_size x n_items x embed_dim]
            mask : [batch_size x n_items] : Bool Tensor

        ** num_items : TOTAL number of items:
            eg : 3 dogs, 5 books, 2 notebooks --> n_items = 3+5+2 = 10
        """
        batch_size, n_items, _ = node_embedding.size()
        query = self.W_q(context)       #[batch x hidden_dim]
        key = self.W_k(node_embedding)      #[batch x seq_len x hidden_dim]
        value = self.W_v(node_embedding)    #[batch x seq_len x hidden_dim]

        qk = torch.einsum("ik, ijk -> ij", query, key)      # [batch x n_items]

        qk = self.clip_constant * torch.tanh(qk)
        # hidden_dim = d_k since we have only one head in this layer.
        if mask is not None:
            qk[mask] = -1000000.0

        ret = torch.softmax(qk, -1)        # [batch x n_items]
        return ret


class Offline_Att_Solver (nn.Module):
    """
    Offline solver:
        모든 item을 한꺼번에 받아서 각각의 우선순의를 output한다.
    """
    def __init__(self, embed_dim, hidden_dim, seq_len, n_head = 4, C = 10, use_cuda = False):
        super(Offline_Att_Solver, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = C          # For clipping the tanh.
        self.use_cuda = use_cuda

        self.node_emb = NodeEmbedding(embed_dim)
        self.encoder = Encoder(embed_dim, n_head)
        self.glimpse = Glimpse(embed_dim, hidden_dim, n_head)
        self.pointer = Pointer(embed_dim, hidden_dim, C)


    def forward(self, inputs):
        """
        Args:
            inputs : [batch x num_items x dimension +1] in knapsack problem.
            where 2 means values and weights.

        ** num_items : TOTAL number of items:
            eg : 3 dogs, 5 books, 2 notebooks --> n_items = 3+5+2 = 10

        Return :
            The probability distribution of each items selected.
        """

        batch_size = inputs.size(0)
        num_items = inputs.size(1)

        embedded = self.node_emb(inputs)
        h = self.encoder(embedded)      #[batch_size x num_items x embed_dim]
        context = h.mean(dim = 1)            # [batch_size x embed_dim]

        order_list = []
        logprobs_list = []
        first_chosen_hs = None
        mask = torch.zeros(batch_size, num_items, dtype = torch.bool)

        for i in range(num_items):
            new_query = self.glimpse(context, embedded, mask)
            probability = self.pointer(new_query, embedded, mask)
            distribution = Categorical(probability)
            sample = distribution.sample()                  # batch_size
            logprobs = distribution.log_prob(sample)        # batch_size
            order_list.append(sample)
            logprobs_list.append(logprobs)

            # We update current context h_C^(N) to another context vector, based on chosen sample.
            tmp = sample.unsqueeze(1).unsqueeze(2).repeate(1, 1, self.embed_dim)
            if first_chosen_hs is None:
                first_chosen_hs = 0
            next_context = h[[i for i in range(batch_size)], sample]        # batch_size x embed_dim
            mask[[i for i in range(batch_size)], sample] = True
            context = next_context

        return torch.stack(order_list, 1), torch.stack(logprobs_list, 1)        # Both : batch_size x num_items
