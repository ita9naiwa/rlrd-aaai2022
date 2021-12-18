import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, activation="relu"):
        super(MLP, self).__init__()
        self.affine = nn.Linear(input_dim, output_dim)
        if activation.lower() == "tanh":
            self.activation = torch.tanh
        elif activation.lower() == "relu":
            self.activation = torch.relu
        else:
            raise NotImplementedError("Customize here")
        # self.normalize = nn.BatchNorm1d(output_dim)

    def forward(self, input):
        output = self.activation(self.affine(input))
        with torch.no_grad():
            omax = torch.max(output)
            omin = torch.min(output)
            output /= (omax - omin)
        return output


class Embedding(nn.Sequential):
    def __init__(self, 
                 input_dim, output_dim, n_layer=1):
        layers = []
        for _ in range(n_layer):
            layers.append(MLP(input_dim, output_dim))
        super(Embedding, self).__init__(*layers)


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads=4, ff_dim=256):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.mha = torch.nn.MultiheadAttention(embed_dim, n_heads)
        self.embed = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        _1 = x + self.mha(x, x, x)[0]
        _1 = _1.permute(1, 0, 2)
        _2 = _1 + self.embed(_1)
        return _2.squeeze(0)


class Attention_Score(nn.Module):
    """
    Calculate the relevence between query and keys.
    This class only calculate the attention score.
    """
    def __init__(self, hidden_size, query_size, use_softmax=False):
        super(Attention_Score, self).__init__()
        self.use_softmax = use_softmax
        self.W_query = nn.Linear(query_size, hidden_size, bias=True)
        self.W_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        V = torch.normal(torch.zeros(hidden_size, 1), 0.0001)
        self.V = nn.Parameter(V)

    def forward(self, query, keys):
        """
        Args:
            query : [query_size]
            keys : [num_items x hidden_size]
        Returns:
            return the attention score.

        For example, keys = [x1, x2, ... xN] --> return : [a1, a2, .. , aN]
        where ai is real number.
        """
        query = self.W_query(query)     # [hidden_size]
        _keys = self.W_ref(keys)        # [num_items x hidden_size]
        m = torch.tanh(query + _keys)
        att_score = torch.matmul(m, self.V)     # [seq_len]
        if self.use_softmax:
            logits = torch.softmax(att_score, dim=0)
        else:
            logits = att_score

        return logits           # [num_items]


class Attention(nn.Module):
    def __init__(self, hidden_size, query_size):
        super(Attention, self).__init__()
        self.attention = Attention_Score(hidden_size, query_size, use_softmax=True)

    def forward(self, query, keys):
        """
        Args:
            query : [query_size]
            keys : [num_items x hidden_size]
        Returns:
            Attention vector

        query = q, keys = [x1, x2, .. , xN]
            --> Return : a1x1 + a2x2 + ... + aNxN
                         where ai determined by previous Attention instance.
        """
        hidden_dim = keys.size(1)
        att_score = self.attention(query, keys).view(1, -1)     # [num_items]
        keys = keys.view(-1, hidden_dim)
        ret = torch.matmul(att_score, keys)     # [hidden_size]
        return ret


class Att_Policy(nn.Module):
    def __init__(self, dimension, embed_dim, use_cuda=False, dqn=False, ret_distribution=False, ret_embedded_vector=False):
        """
        Args:
            dimension : The dimension of item and knapsack capacity.
            embed_dim : The embedding dimension of our attention model
        """
        super(Att_Policy, self).__init__()
        self.dimension = dimension
        self.embed_dim = embed_dim
        self.ret_distribution = ret_distribution
        self.item_embedding = MLP(dimension + 10, embed_dim)       # +1 for the value of item.
        self.attention_layer = AttentionLayer(embed_dim=embed_dim)
        self.att_to_item = Attention(embed_dim, embed_dim)
        self.global_item_attention = Attention_Score(embed_dim, embed_dim * 2)       # query size: embed_dim * 3
        self.use_cuda = use_cuda
        self.ret_embedded_vector = ret_embedded_vector
        # Last_item : We use previously chosen item to select a new item.
        # 방금 전에 마지막으로 고른 아이템을 query삼아서 다음 time step의 아이템을 고를 것이다.
        self.last_item = torch.normal(torch.zeros(self.embed_dim), 0.001)

        self.log_probs = []
        self.cumulated_distributions = []
        if self.use_cuda:
            self.last_item.to("cuda:0")
        else:
            pass

    def reset(self, dqn=False, num_items=0):
        if self.use_cuda:
            self.last_item = torch.normal(torch.zeros(self.embed_dim), 0.001).to("cuda:0")
        else:
            self.last_item = torch.normal(torch.zeros(self.embed_dim), 0.001)
        self.log_probs = []

        if dqn:
            self.dqn_linear = nn.Linear(num_items, num_items)

        if self.ret_distribution:
            self.cumulated_distributions = []

    def __get_item_attention(self, input, query):
        return self.att_to_item(input, query)

    # def __get_knapsack_attention(self, input, query):
    #     return self.att_to_knapsack(input, query)

    def __embedding(self, items):
        it = self.item_embedding(items)
        # kn = self.knapsack_embedding(knapsack)
        return it

    def __input_transform(self, items, knapsack):
        """
        :param items: [num_items x (1 + dimension)]
        :param knapsack: [1 x dimension]
        """
        value = items[:, 0].unsqueeze(1)
        weight = items[:, 1:]
        util = weight / knapsack
        meanutil = torch.mean(util, dim=-1, keepdim=True)
        maxutil, _ = torch.max(util, dim=-1, keepdim=True)
        minutil, _ = torch.min(util, dim=-1, keepdim=True)  # ~util : [num_items x 1]
        a1 = value / meanutil   # [num_items x 1]
        a2 = value / maxutil   # [num_items x 1]
        a3 = value / minutil   # [num_items x 1]
        a4 = meanutil / maxutil
        a5 = meanutil / minutil
        a6 = maxutil / minutil
        ret = torch.cat([items, meanutil, maxutil, minutil,
                         a1, a2, a3, a4, a5, a6], dim=-1)
        return ret

    def forward(self, items, knapsacks,
                allocable_items, allocable_knapsacks,
                argmax=False, guide=False, return_score=False):
        """
        Choose one item among the allocable items
        Args:
            items : [num_items x (dimension + 1)] : +1 comes from the value of items.
                    dimension is for the weight.
            knapsacks : [num_knapsacks x dimension]
            allocable_items : list of integer, indicating the index of allocable items.
        """
        _items = items.detach()
        items = self.__input_transform(items, knapsacks)
        num_items = items.size(0)
        num_knapsacks = knapsacks.size(0)
        Ei = self.__embedding(items)       # Ei : [num_items x embed_dim], Ek : [num_knapsack x embed_dim]
        Gi = self.__get_item_attention(self.last_item, Ei)        # Gi : [1 x embed_dim]
        # Gk = self.__get_knapsack_attention(self.last_item, Ek)        # Gk : [1 x embed_dim]
        global_vector1 = torch.cat([self.last_item.unsqueeze(0), Gi], -1)
        item_logits = self.global_item_attention(global_vector1, Ei).squeeze()       # [num_items]
        mask = []
        for i in range(num_items):
            if i not in allocable_items:
                mask.append(i)
        item_logits[mask] = -1e8
        item_softmax = torch.softmax(item_logits, dim=0)
        if self.ret_distribution:
            self.cumulated_distributions.append(item_softmax)
        item_sampler = Categorical(item_softmax)
        if guide:
            util = torch.mean(_items[:, 1:] / knapsacks, dim=-1)
            util = _items[:, 0] / util  # [num_items]
            util[mask] = 0
            selected_item = torch.argmax(util)
        elif argmax:
            selected_item = torch.argmax(item_softmax)
        else:
            selected_item = item_sampler.sample()
        item_logprob = item_sampler.log_prob(selected_item)

        item_idx = selected_item.item()
        allocable_knapsack = allocable_knapsacks[item_idx]
        mask = []
        for i in range(num_knapsacks):
            if i not in allocable_knapsack:
                mask.append(i)

        # embedded_selected_job = Ei[selected_item]
        # global_job2 = self.__get_item_attention(embedded_selected_job, Ei)
        # global_kn2 = self.__get_knapsack_attention(embedded_selected_job, Ek)
        # global_vector2 = torch.cat([embedded_selected_job.unsqueeze(0), global_job2, global_kn2], -1)
        # knapsack_logits = self.global_knapsack_attention(global_vector2, Ek)
        # knapsack_logits = 5 * torch.tanh(knapsack_logits)     # Bound the logits
        # knapsack_logits[mask] = -1e8
        # knapsack_softmax = torch.softmax(knapsack_logits, dim=0).squeeze(1)
        # knapsack_sampler = Categorical(knapsack_softmax)

        if return_score:
            return item_logits

        # if argmax:
        #     selected_knapsack = torch.argmax(knapsack_softmax)
        # else:
        #     selected_knapsack = knapsack_sampler.sample()
        # knapsack_logprob = knapsack_sampler.log_prob(selected_knapsack)

        # log_prob = item_logprob + knapsack_logprob
        log_prob = item_logprob
        self.log_probs.append(log_prob)         # Store the log probability to policy gradient.
        self.last_item = self.item_embedding(items[item_idx])        # Update the last selected item.
        if self.ret_embedded_vector:
            return (int(selected_item.detach().numpy())), log_prob, Ei
        if self.use_cuda:
            return (int(selected_item.detach().cpu().numpy())), log_prob
        else:
            return (int(selected_item.detach().numpy())), log_prob

    def load_finetune_checkpoint(self, path):

        m = torch.load(path).state_dict()
        model_dict = self.state_dict()
        for k in m.keys():
            if 'item_embedding' in k:
                continue
            if k in model_dict:
                pname = k
                pval = m[k]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)

        self.load_state_dict(model_dict)

        for name, param in self.named_parameters():
            if not name.startswith("item_embedding"):
                param.requires_grad_(False)

class Encoder_Distilation(nn.Module):
    def __init__(self, dimension, embed_dim, use_cuda = False):
        super(Encoder_Distilation, self).__init__()
        self.dimension = dimension
        self.embed_dim = embed_dim
        self.item_embedding = Embedding(dimension + 10, embed_dim)
        self.knapsack_embedding = Embedding(dimension, embed_dim)
        self.att_to_item = Attention(embed_dim, embed_dim)
        self.att_to_knapsack = Attention(embed_dim, embed_dim)
        self.global_item_attention = Attention_Score(embed_dim, embed_dim * 2)  # query size: embed_dim * 3
        # self.global_knapsack_attention = Attention_Score(embed_dim, embed_dim * 3)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.learnable_parameter = torch.normal(torch.zeros(self.embed_dim), 0.0001).to("cuda:0")
        else:
            self.learnable_parameter = torch.normal(torch.zeros(self.embed_dim), 0.0001)

        self.use_cuda = use_cuda

    def __input_transform(self, items, knapsack):
        """
                :param items: [num_items x (1 + dimension)]
                :param knapsack: [1 x dimension]
                """
        value = items[:, 0].unsqueeze(1)
        weight = items[:, 1:]
        util = weight / knapsack
        meanutil = torch.mean(util, dim=-1, keepdim=True)
        maxutil, _ = torch.max(util, dim=-1, keepdim=True)
        minutil, _ = torch.min(util, dim=-1, keepdim=True)  # ~util : [num_items x 1]
        a1 = value / meanutil  # [num_items x 1]
        a2 = value / maxutil  # [num_items x 1]
        a3 = value / minutil  # [num_items x 1]
        a4 = meanutil / maxutil
        a5 = meanutil / minutil
        a6 = maxutil / minutil
        ret = torch.cat([items, meanutil, maxutil, minutil,
                         a1, a2, a3, a4, a5, a6], dim=-1)
        return ret

    def reset(self):
        pass

    def __get_item_attention(self, input, query):
        return self.att_to_item(input, query)

    def __get_knapsack_attention(self, input, query):
        return self.att_to_knapsack(input, query)

    def __embedding(self, items):
        it = self.item_embedding(items)
        return it

    def forward(self, items, knapsacks):
        _items = items.detach()
        items = self.__input_transform(items, knapsacks)
        Ei = self.__embedding(items)  # Ei : [num_items x embed_dim]
        Gi = self.__get_item_attention(self.learnable_parameter, Ei)  # Gi : [1 x embed_dim]
        # Gk = self.__get_knapsack_attention(self.learnable_parameter, Ek)  # Gk : [1 x embed_dim]
        global_vector = torch.cat([self.learnable_parameter.unsqueeze(0), Gi], -1)
        item_logits = self.global_item_attention(global_vector, Ei)  # [num_items x 1]
        return item_logits.view(1, -1)


class Linear_Policy(nn.Module):
    def __init__(self, dimension, use_cuda=False):
        """
            If policy is True, it acts as a policy.
            If policy is False, this is for the soft-rank based training model.
        """
        super(Linear_Policy, self).__init__()
        self.item_layer = nn.Linear(dimension + 1, 1)
        self.knapsack_layer = nn.Linear(dimension, 1)
        # torch.nn.init.normal_(self.item_layer.weight.data, 0, 0.0001)
        torch.nn.init.uniform_(self.item_layer.weight.data, 0, 0.001)
        torch.nn.init.normal_(self.knapsack_layer.weight.data, 0, 0.001)

        self.use_cuda = use_cuda
        self.log_probs = []

    def reset(self):
        self.log_probs = []

    def forward(self, items, knapsacks,
                allocable_items, allocable_knapsacks,
                argmax=False):
        num_items = items.size(0)
        num_knapsacks = knapsacks.size(0)
        item_logits = self.item_layer(items).squeeze()        # [num_items]
        mask = []
        for i in range(num_items):
            if i not in allocable_items:
                mask.append(i)
        # item_logits = args.clipping_const * torch.tanh(item_logits)      # Just for bounding.
        item_logits[mask] = -1e8
        item_softmax = torch.softmax(item_logits, dim=0)
        item_sampler = Categorical(item_softmax)
        if argmax:
            selected_item = torch.argmax(item_softmax)
        else:
            selected_item = item_sampler.sample()

        item_logprob = item_sampler.log_prob(selected_item)

        if num_knapsacks != 1:
            knapsack_logits = self.knapsack_layer(knapsacks).squeeze()  # [num_knapsacks]
        else:
            knapsack_logits = self.knapsack_layer(knapsacks)

        item_idx = selected_item.item()
        allocable_knapsack = allocable_knapsacks[item_idx]
        mask = []
        for i in range(num_knapsacks):
            if i not in allocable_knapsack:
                mask.append(i)

        # knapsack_logits = args.clipping_const * torch.tanh(knapsack_logits)
        knapsack_logits[mask] = -1e8
        knapsack_softmax = torch.softmax(knapsack_logits, dim=0)
        knapsack_sampler = Categorical(knapsack_softmax)

        if argmax:
            selected_knapsack = torch.argmax(knapsack_softmax)
        else:
            selected_knapsack = knapsack_sampler.sample()
        knapsack_logprob = knapsack_sampler.log_prob(selected_knapsack)
        log_prob = item_logprob + knapsack_logprob
        self.log_probs.append(log_prob)

        if self.use_cuda:
            return (int(selected_item.detach().cpu().numpy())),\
                   (int(selected_knapsack.detach().cpu().numpy())), log_prob
        else:
            return (int(selected_item.detach().numpy())),\
                   (int(selected_knapsack.detach().numpy())), log_prob


class Linear_Distilation(nn.Module):
    def __init__(self, dimension):
        super(Linear_Distilation, self).__init__()
        self.item_layer = nn.Linear(dimension + 1, 1)
        self.knapsack_layer = nn.Linear(dimension, 1)

    def reset(self):
        pass

    def forward(self, items, knapsacks):
        """
        Args:
             items : [num_items x dimension]
             knapsacks : [num_knapsacks x dimension]
        """
        item_score = self.item_layer(items)      # [num_items x 1]
        knapsack_score = self.knapsack_layer(knapsacks)      # [num_knapsacks x 1]
        score = torch.matmul(
            knapsack_score, item_score.view(1, -1)
        )        # [num_knapsacks x num_items]
        return score