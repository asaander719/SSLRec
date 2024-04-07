import math
import torch as t
from torch import nn
from torch.nn import init
import dgl.function as fn
from config.configurator import configs
import torch.nn.functional as F
import torch
import copy

class SpAdjEdgeDrop(nn.Module):
    def __init__(self, resize_val=False):
        super(SpAdjEdgeDrop, self).__init__()
        self.resize_val = resize_val

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (t.rand(edgeNum) + keep_rate).floor().type(t.bool)
        newVals = vals[mask] / (keep_rate if self.resize_val else 1.0)
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class NodeDrop(nn.Module):
    def __init__(self):
        super(NodeDrop, self).__init__()

    def forward(self, embeds, keep_rate):
        if keep_rate == 1.0:
            return embeds
        data_config = configs['data']
        node_num = data_config['user_num'] + data_config['item_num']
        mask = (t.rand(node_num) + keep_rate).floor().view([-1, 1])
        return embeds * mask


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = "both"
        if weight:
            self.weight = nn.Parameter(t.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)  # outdegree of nodes
            norm = t.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)  # (n, 1)
            norm = t.reshape(norm, shp)  # (n, 1)
            # feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = t.matmul(feat, weight)
            feat = feat * norm
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            if weight is not None:
                rst = t.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = t.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = t.reshape(norm, shp)
            rst = rst * norm
        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layer = GraphConv(in_feats, n_hidden, weight=False, activation=activation)

    def forward(self, features):
        h = features
        h = self.layer(self.g, h)
        return h


def message_func(edges):
    return {'m': edges.src['n_f'] + edges.data['e_f']}

class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 bias=False,
                 activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            init.xavier_uniform_(self.u_w)
            init.xavier_uniform_(self.v_w)
        self._activation = activation

    def forward(self, graph, u_f, v_f, e_f):
        with graph.local_scope():
            if self.weight:
                u_f = t.mm(u_f, self.u_w)
                v_f = t.mm(v_f, self.v_w)
            node_f = t.cat([u_f, v_f], dim=0)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.edata['e_f'] = e_f
            graph.update_all(message_func=message_func, reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.d_k = hidden_size // num_heads
        self.n_h = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def _cal_attention(self, query, key, value, mask=None, dropout=None):
        scores = t.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return t.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.n_h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self._cal_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        # return x + self.dropout(sublayer(x))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feed_forward_size, dropout_rate): #self.emb_size * 4
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout_rate)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, d_ff=feed_forward_size, dropout=dropout_rate)
        self.input_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.output_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, item_num, emb_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token_emb = nn.Embedding(item_num, emb_size, padding_idx=0)
        self.position_emb = nn.Embedding(max_len, emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

    def forward(self, batch_seqs):
        batch_size = batch_seqs.size(0)
        pos_emb = self.position_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.token_emb(batch_seqs) + pos_emb
        return self.dropout(x)


class DGIEncoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(DGIEncoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, activation)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = t.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class DGIDiscriminator(nn.Module):
    def __init__(self, n_hidden):
        super(DGIDiscriminator, self).__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(t.empty(n_hidden, n_hidden)))
        self.loss = nn.BCEWithLogitsLoss(reduction='none')  # combines a Sigmoid layer and the BCELoss

    def forward(self, node_embedding, graph_embedding, corrupt=False):
        score = t.sum(node_embedding * graph_embedding, dim=1)

        if corrupt:
            res = self.loss(score, t.zeros_like(score))
        else:
            res = self.loss(score, t.ones_like(score))
        return res


# class TransformerEncoder(nn.Module):
#     def __init__(
#         self,
#         n_layers=2,
#         num_heads=2,
#         hidden_size=64,
#         inner_size=256, 
#         hidden_dropout_prob=0.5,
#         attn_dropout_prob=0.5,
#         hidden_act='gelu',
#         layer_norm_eps=1e-12,
#         multiscale=False,
#         scales=None
#     ):

#         super(TransformerEncoder, self).__init__()
#         layer = TransformerLayer_LSY(
#             num_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, multiscale=multiscale, scales=scales
#         )
#         self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

#     def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
#         all_encoder_layers = []
#         for layer_module in self.layer:
#             hidden_states = layer_module(hidden_states, attention_mask)
#             if output_all_encoded_layers:
#                 all_encoder_layers.append(hidden_states)
#         if not output_all_encoded_layers:
#             all_encoder_layers.append(hidden_states)
#         return all_encoder_layers

# class TransformerLayer_LSY(nn.Module):
#     """
#     One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

#     Args:
#         hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
#         attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

#     Returns:
#         feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
#                                            is the output of the transformer layer.

#     """

#     def __init__(
#         self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
#         layer_norm_eps,
#         multiscale=False,
#         scales=None
#     ):
#         super(TransformerLayer_LSY, self).__init__()
#         if multiscale:
#             self.multi_head_attention = MultiScaleAttention_LSY(
#             scales, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
#         )
#         else:
#             self.multi_head_attention = MultiHeadAttention_LSY(
#             n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
#         )
#         self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

#     def forward(self, hidden_states, attention_mask):
#         attention_output = self.multi_head_attention(hidden_states, attention_mask)
#         feedforward_output = self.feed_forward(attention_output)
#         return feedforward_output

# class MultiScaleAttention_LSY(nn.Module):
#     """
#     a set of attention of different granularities
#     """

#     def __init__(self, scales, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
#         super().__init__()
#         assert hidden_size % n_heads == 0
#         self.d_k = hidden_size // n_heads
#         self.num_heads = n_heads
#         self.scale_1 = scales[1]
#         self.scale_2 = scales[2]
#         self.out_fc = nn.Linear(200+200//self.scale_1+200//self.scale_2, 200)

#         # self.attention1 = LinearMultiheadAttention(hidden_dim, num_heads, seq_len=200, proj_k=args.linear_size)
#         self.attention1 = LinearAttention(n_heads, scales[0], hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
#         self.attention2 = MultiHeadAttention_LSY(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)

#     def forward(self, input_tensor, attention_mask):
#         batch_size = input_tensor.size(0)
#         seq_length = input_tensor.size(1)

#         # 2) multi scale attention

#         # linear attention over whole sequence
#         # b, num_heads, seq_length, dim//num_heads
#         x, linear_attn_weight = self.attention1(input_tensor, attention_mask)
#         # x, _ = self.attention1(query, key, value, key_padding_mask=mask.squeeze())
#         scale_outputs = []
#         scale_outputs.append(torch.reshape(x, [batch_size, seq_length, self.num_heads*self.d_k]))
#         next_input = torch.mean(input_tensor.reshape(batch_size, self.scale_1, seq_length//self.scale_1, self.num_heads*self.d_k), dim=1)

#         # attention over 1/scale_1 sequence
#         x = self.attention2(next_input, None)
#         scale_outputs.append(torch.reshape(x, [batch_size, seq_length//self.scale_1, self.num_heads*self.d_k]))
#         # next_input = torch.mean(x.reshape(batch_size, self.scale_2//self.scale_1, seq_length//self.scale_2, self.num_heads*self.d_k), dim=1)
#         next_input = torch.mean(input_tensor.reshape(batch_size, self.scale_2, seq_length//self.scale_2, self.num_heads*self.d_k), dim=1)


#         # attention over 1/scale_2 sequence
#         x = self.attention2(next_input, None)
#         scale_outputs.append(torch.reshape(x, [batch_size, seq_length//self.scale_2, self.num_heads*self.d_k]))

#         output = torch.cat(scale_outputs, dim=1)
#         output = torch.transpose(output, 1, 2)
#         output = self.out_fc(output)
#         output = torch.transpose(output, 1, 2)

#         return output
#         # return output, [linear_attn_weight.mean(1), pool_attn_weight_1.mean(1), pool_attn_weight_2.mean(1)]

# class MultiHeadAttention_LSY(nn.Module):
#     """
#     Multi-head Self-attention layers, a attention score dropout layer is introduced.

#     Args:
#         input_tensor (torch.Tensor): the input of the multi-head self-attention layer
#         attention_mask (torch.Tensor): the attention mask for input tensor

#     Returns:
#         hidden_states (torch.Tensor): the output of the multi-head self-attention layer

#     """

#     def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
#         super(MultiHeadAttention_LSY, self).__init__()
#         if hidden_size % n_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (hidden_size, n_heads)
#             )

#         self.num_attention_heads = n_heads
#         self.attention_head_size = int(hidden_size / n_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(hidden_size, self.all_head_size)
#         self.key = nn.Linear(hidden_size, self.all_head_size)
#         self.value = nn.Linear(hidden_size, self.all_head_size)

#         self.attn_dropout = nn.Dropout(attn_dropout_prob)

#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.LayerNorm = nn.LayerNorm(hidden_size)#, eps=layer_norm_eps)
#         self.out_dropout = nn.Dropout(hidden_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, input_tensor, attention_mask):
#         mixed_query_layer = self.query(input_tensor)
#         mixed_key_layer = self.key(input_tensor)
#         mixed_value_layer = self.value(input_tensor)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # [batch_size heads seq_len seq_len] scores
#         # [batch_size 1 1 seq_len]
#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.

#         attention_probs = self.attn_dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         hidden_states = self.dense(context_layer)
#         hidden_states = self.out_dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states

# class LinearAttention(nn.Module):
#     """
#     compute linear attention using projection E and F.
#     """
#     def __init__(self, num_heads, linear_size, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
#         super(LinearAttention, self).__init__()
#         self.E = nn.Linear(200, linear_size)
#         self.F = nn.Linear(200, linear_size)
#         self.W_V = nn.Linear(hidden_size, hidden_size)
#         self.W_K = nn.Linear(hidden_size, hidden_size)
#         self.W_Q = nn.Linear(hidden_size, hidden_size)

#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.n_heads = num_heads
#         self.d_k = hidden_size // num_heads
#         self.attn_dropout = nn.Dropout(p=attn_dropout_prob)
#         self.out_dropout = nn.Dropout(p=hidden_dropout_prob)
#         self.LayerNorm = nn.LayerNorm(hidden_size)#, eps=layer_norm_eps)
    
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.n_heads, self.d_k)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, input_tensor, mask=None):
#         key = self.W_K(input_tensor)
#         value = self.W_V(input_tensor)
#         query = self.W_Q(input_tensor)

#         key = self.transpose_for_scores(key)
#         query = self.transpose_for_scores(query)
#         value = self.transpose_for_scores(value)

#         # b, num_heads, l, d/num_heads
#         if mask is not None:
#             mask = mask[:,0:1,:].unsqueeze(-1)
#             # b, 1, l, 1
#             key = key*mask
#             value = value*mask

#         value = self.E(value.transpose(2, 3)).transpose(2, 3)
#         key = self.F(key.transpose(2, 3)).transpose(2, 3)

#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(query.size(-1))

#         p_attn = F.softmax(scores, dim=-1)

#         p_attn = self.attn_dropout(p_attn)

#         context_layer = torch.matmul(p_attn, value)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.d_k*self.n_heads,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         hidden_states = self.dense(context_layer)
#         hidden_states = self.out_dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states, p_attn

# class FeedForward(nn.Module):
#     """
#     Point-wise feed-forward layer is implemented by two dense layers.

#     Args:
#         input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

#     Returns:
#         hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

#     """

#     def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
#         super(FeedForward, self).__init__()
#         self.dense_1 = nn.Linear(hidden_size, inner_size)
#         self.intermediate_act_fn = self.get_hidden_act(hidden_act)

#         self.dense_2 = nn.Linear(inner_size, hidden_size)
#         self.LayerNorm = nn.LayerNorm(hidden_size) #, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(hidden_dropout_prob)

#     def get_hidden_act(self, act):
#         ACT2FN = {
#             "gelu": self.gelu,
#             # "relu": fn.relu,
#             "swish": self.swish,
#             "tanh": torch.tanh,
#             "sigmoid": torch.sigmoid,
#         }
#         return ACT2FN[act]

#     def gelu(self, x):
#         """Implementation of the gelu activation function.

#         For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

#             0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

#         Also see https://arxiv.org/abs/1606.08415
#         """
#         return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#     def swish(self, x):
#         return x * torch.sigmoid(x)

#     def forward(self, input_tensor):
#         hidden_states = self.dense_1(input_tensor)
#         hidden_states = self.intermediate_act_fn(hidden_states)

#         hidden_states = self.dense_2(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states

class HGNN(nn.Module):
    def __init__(self, n_hid, dropout=0.2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        # self.out_fc = nn.Linear(n_hid, n_hid)

    def forward(self, x, G):
        x1 = self.hgc1(x, G)
        # x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        return x1
        # x2 = self.hgc2(x1, G)
        # # x2 = F.relu(x2)
        # x2 = F.dropout(x2, self.dropout, training=self.training)
        # return ((x1+x2)/2)
        # # return self.out_fc((x1+x2)/2)

class HGNN_conv(nn.Module):
    def __init__(self, n_hid, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(n_hid, n_hid))
        nn.init.normal_(self.weight, std=0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_hid))
            nn.init.normal_(self.bias, std=0.02)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)

        # 激活函数
        self.activation = nn.ReLU()
        self.activation_out = nn.Sigmoid() #nn.Tanh() ##考虑noise负分的情况用Tanh

    def forward(self, x):
        if x.size(-1) != self.input_size:
            x = x.view(-1, self.input_size) 
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        x = self.activation_out(x)
        return x
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super(MLP, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers

#         self.hidden_layers = nn.ModuleList()
#         self.hidden_layers.append(nn.Linear(input_size, hidden_size))
#         for _ in range(num_layers - 1):
#             self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
#         # 输出层
#         self.output_layer = nn.Linear(hidden_size, output_size)
#         # 激活函数
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         if x.size(-1) != self.input_size:
#             x = x.view(-1, self.input_size) 
#         for layer in self.hidden_layers:
#             x = self.activation(layer(x))
#         x = self.output_layer(x)
#         x = torch.softmax(x, dim=1)
#         return x

class SelfAttentionModel(nn.Module): # for calculate the contribution of each item in the sequence
    def __init__(self, hidden_size, output_size, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedded_seq):
        # Multi-head self-attention
        attn_output, attn_output_weights = self.attention(embedded_seq, embedded_seq, embedded_seq)  # b,l,h ; b, l,l 
        # Apply attention weights
        weighted_seq = attn_output.mean(dim=1)  # Mean pooling over attention output
        output = self.fc(weighted_seq)
        return F.softmax(output, dim=1) #output
        # # the softmax importance score of each item in the sequence
        # Contribution Scores
        # 3/18
        # scores = self.softmax(attn_output.sum(dim=-1))  # (batch_size, seq_len) #实验证明不好用！！
        # return scores
'''attn_output_weights是自注意力机制的输出 它表示了每个输入元素对于输出元素的贡献程度。直接使用attn_output_weights作为每个item的贡献得分。
然而 attn_output_weights的形状是(batch_size, seq_len, seq_len) 它表示了序列中每个元素对于每个元素的贡献。如果想要得到每个元素的总贡献得分
可能需要对attn_output_weights进行一些操作 例如求和或者取平均。
另一方面 attn_output是自注意力机制的输出 它已经考虑了每个元素的贡献。因此对attn_output求和并应用softmax函数可以得到每个元素的贡献得分
这可能更直接一些。'''