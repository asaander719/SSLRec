import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.model_utils import TransformerLayer, TransformerEmbedding, TransformerEncoder, HGNN
from models.base_model import BaseModel
from config.configurator import configs

def sim(z1: torch.Tensor, z2: torch.Tensor): #cosine sim
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.matmul(z1, z2.permute(0,2,1))

class LSY(BaseModel):
    def __init__(self, data_handler):
        super(LSY, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.hidden_size = configs['model']['hidden_size'] # same as embedding_size
        self.inner_size = configs['model']['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = configs['model']['hidden_dropout_prob']
        self.attn_dropout_prob = configs['model']['attn_dropout_prob']
        self.hidden_act = configs['model']['hidden_act']
        self.layer_norm_eps = 1e-12
        self.mask_ratio = configs['model']['mask_ratio']
        self.loss_type = configs['model']['loss_type']
        self.hglen = configs['model']['hyper_len']
        self.enable_hg = configs['tune']['enable_hg']
        self.enable_ms = configs['tune']['enable_ms']
        self.dataset = configs['data']['name']
        self.max_seq_length = configs['model']['max_seq_len']

        # self.buy_type = dataset.field2token_id["item_type_list"]['0']

        self.mask_token = self.item_num 
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num +1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)  # add mask_token at the last

        if self.enable_ms:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                num_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=True,
                scales=configs['model']["scales"]
            )
        else:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                num_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=False
            )

        self.hgnn_layer = HGNN(self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size) #, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.hg_type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.gating_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # self.position_range = 3
        self.sim_threshold = 0.1
        self.max_sim_length =14
        
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        # nn.init.normal_(self.gating_bias, std=0.02)
        nn.init.normal_(self.gating_weight, std=0.02)
        nn.init.normal_(self.metric_w1, std=0.02)
        nn.init.normal_(self.metric_w2, std=0.02)

        if self.dataset == "retail_beh":
            self.sw_before = 10
            self.sw_follow = 6
        elif self.dataset == "ijcai_beh":
            self.sw_before = 30
            self.sw_follow = 18
        elif self.dataset == "tmall_beh":
            self.sw_before = 20
            self.sw_follow = 12
        elif self.dataset == "ml-20m":
            self.sw_before = 10
            self.sw_follow = 6
        elif self.dataset == "beauty":
            self.sw_before = 10
            self.sw_follow = 6

        self.hypergraphs = dict()
        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['BPR', 'CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-scale attention."""
        if self.enable_ms:
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1)
            return extended_attention_mask
        else:
            """Generate bidirectional attention mask for multi-head attention."""
            attention_mask = (item_seq > 0).long()
            #The result is a tensor of 0s and 1s, where 1 indicates a valid item, and 0 indicates a padding or invalid item.
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
            # bidirectional mask
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 
            #after softmax function, these positions will have zero weight in the attention mechanism.
            return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def _transform_train_seq(self, item_seq, type_seq, last_buy):
        last_buy = last_buy.tolist()
        device = item_seq.device
        batch_size = item_seq.size(0)
        # item_seq, type_seq = self.sample_sequence(item_seq, type_seq)
        # zero_padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        # item_seq = torch.cat((item_seq, zero_padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        # if type_seq is not None:
        #     type_seq = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        n_objs = (torch.count_nonzero(item_seq, dim=1)+1).tolist()
        for batch_id in range(batch_size):
            n_obj = n_objs[batch_id]
            item_seq[batch_id][n_obj-1] = last_buy[batch_id]
            if type_seq is not None:
                type_seq[batch_id][n_obj-1] = self.buy_type
        sequence_instances = item_seq.cpu().numpy().tolist()
        if type_seq is not None:
            type_instances = type_seq.cpu().numpy().tolist()
        masked_item_sequence = []
        pos_items = []
        masked_index = []

        for instance_idx, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if index_id == n_objs[instance_idx]-1:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    if type_seq is not None:
                        type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    if type_seq is not None:
                        type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
            #The list of masked items (pos_items) and their indices (masked_index) are padded to a fixed length (self.mask_item_length).
            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        #pos_items.size() = torch.Size([64, 40])
        # [B mask_len] #[[ 0,     0,     0,  ...,     0, 16308,  1998],[    0,     0,     0,  ...,  7592,  3486,  8531]]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        if type_seq is not None:
            type_instances = torch.tensor(type_instances, dtype=torch.long, device=device).view(batch_size, -1)
        else:
            type_instances = None
        return masked_item_sequence, pos_items, masked_index, type_instances

    def _transform_test_seq(self, item_seq, item_seq_len, item_type):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        # item_seq, item_type = self.sample_sequence(item_seq, item_type)
        
        # padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        # item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        # if item_type is not None:
        #     item_type = torch.cat((item_type, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq, item_type

    def build_graph(self, item_seq):
        unique_items = torch.unique(item_seq[item_seq != self.mask_token])

        # Initialize adjacency matrix
        adj_matrix = torch.zeros((item_seq.size(0), len(unique_items), len(unique_items)))

        for batch_idx in range(item_seq.size(0)):
            # Add adjacent edges for items in the sequence
            for i in range(len(item_seq[batch_idx]) - 1):
                if item_seq[batch_idx, i] != self.mask_token and item_seq[batch_idx, i + 1] != self.mask_token:
                    idx_i = torch.where(unique_items == item_seq[batch_idx, i])[0]
                    idx_j = torch.where(unique_items == item_seq[batch_idx, i + 1])[0]
                    adj_matrix[batch_idx, idx_i, idx_j] = 1.0

            # Add similarity edges for short sequences
            if len(item_seq[batch_idx]) < self.max_sim_length:
                for i, item_i in enumerate(unique_items):
                    idx_i = torch.where(unique_items == item_i)[0]
                    for j, item_j in enumerate(unique_items):
                        idx_j = torch.where(unique_items == item_j)[0]
                        if self.sim_matrix[idx_i, idx_j] >= self.sim_threshold:
                            adj_matrix[batch_idx, i, j] = self.sim_matrix[idx_i, idx_j]
        return adj_matrix

    def forward(self, item_seq, type_seq, mask_positions_nums=None, session_id=None):
        # if self.dataset == "ml-20m" or self.dataset == "beauty":
        #     session_id, item_seq, last_buy = batch_seqs
        #     type_seq = None
        # else:
        #     session_id, item_seq, type_seq, last_buy = batch_seqs
        item_emb = self.item_embedding(item_seq)
        if type_seq is not None:
            type_embedding = self.type_embedding(type_seq)
            input_emb = item_emb + type_embedding     
        else:
            input_emb = item_emb 
        input_emb = self.LayerNorm(input_emb)
        x_raw = self.dropout(input_emb)  # torch.Size([128, 200, 64])
        extended_attention_mask = self.get_attention_mask(item_seq)
        # x_raw = item_emb 
        x_raw = x_raw * torch.sigmoid(x_raw.matmul(self.gating_weight)+self.gating_bias) # torch.Size([128, 200, 64]) #func. 12?
        # b, l, l #passed through a gating mechanism involving a sigmoid function
        x_m = torch.stack((self.metric_w1*x_raw, self.metric_w2*x_raw)).mean(0) # torch.Size([128, 200, 64])
        item_sim = sim(x_m, x_m) #within the seq # cosine sim
        item_sim[item_sim < 0] = 0.01 # torch.Size([128, 200, 200])
        Gs = self.build_Gs_unique(item_seq, item_sim, self.hglen)

        batch_size = item_seq.shape[0]
        # print(item_seq.size())torch.Size([512, 201])
        seq_len = item_seq.shape[1]
        n_objs = torch.count_nonzero(item_seq, dim=1)
        indexed_embs = list() # for each non-zero object in the sequence.
        for batch_idx in range(batch_size):
            n_obj = n_objs[batch_idx] # l', dim
            indexed_embs.append(x_raw[batch_idx][:n_obj])
        indexed_embs = torch.cat(indexed_embs, dim=0)
        hgnn_embs = self.hgnn_layer(indexed_embs, Gs) #torch.Size([696, 64]) torch.Size([631, 64])
        
        hgnn_take_start = 0
        hgnn_embs_padded = []
        for batch_idx in range(batch_size):
            n_obj = n_objs[batch_idx]
            embs = hgnn_embs[hgnn_take_start:hgnn_take_start+n_obj]
            hgnn_take_start += n_obj # l', dim || padding emb -> l, dim
            padding = torch.zeros((seq_len-n_obj, embs.shape[-1])).to(item_seq.device)
            embs = torch.cat((embs, padding), dim=0)
            pos = (item_seq[batch_idx]==self.mask_token).nonzero(as_tuple=True)[0][0]
            sliding_window_start = pos-self.sw_before if pos-self.sw_before>-1 else 0
            embs[pos] = torch.mean(embs[sliding_window_start:pos], dim=0)
            hgnn_embs_padded.append(embs)
        # b, l, dim
        hgnn_embs = torch.stack(hgnn_embs_padded, dim=0)
        trm_output = self.trm_encoder(hgnn_embs, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1] 
        return output  # [B L H]

    def build_Gs_unique(self, seqs, item_sim, group_len):
        Gs = []
        n_objs = torch.count_nonzero(seqs, dim=1).tolist()
        for batch_idx in range(seqs.shape[0]):
            seq = seqs[batch_idx]
            n_obj = n_objs[batch_idx]
            seq = seq[:n_obj]#.cpu()
            seq_list = seq.tolist()
            unique = torch.unique(seq)
            unique = unique.tolist()
            n_unique = len(unique)

            # multibeh_group = seq.tolist()
            # for x in unique:
            #     multibeh_group.remove(x)
            # multibeh_group = list(set(multibeh_group))
            # try:
            #     multibeh_group.remove(self.mask_token)
            # except:
            #     pass
            
            # l', l'
            seq_item_sim = item_sim[batch_idx][:n_obj, :][:, :n_obj]            
            # l', group_len
            if group_len>n_obj:
                metrics, sim_items = torch.topk(seq_item_sim, n_obj, sorted=False)
            else:
                metrics, sim_items = torch.topk(seq_item_sim, group_len, sorted=False)
            # map indices to item tokens
            sim_items = seq[sim_items]
            row_idx, masked_pos = torch.nonzero(sim_items==self.mask_token, as_tuple=True)
            sim_items[row_idx, masked_pos] = seq[row_idx]
            metrics[row_idx, masked_pos] = 1.0
            # print(sim_items.detach().cpu().tolist())
            # multibeh_group = seq.tolist()
            # for x in unique:
            #     multibeh_group.remove(x)
            # multibeh_group = list(set(multibeh_group))
            # try:
            #     multibeh_group.remove(self.mask_token)
            # except:
            #     pass
            n_edge = n_unique#+len(multibeh_group)
            # hyper graph: n_obj, n_edge
            H = torch.zeros((n_obj, n_edge), device=metrics.device)
            normal_item_indexes = torch.nonzero((seq != self.mask_token), as_tuple=True)[0]
            for idx in normal_item_indexes:
                sim_items_i = sim_items[idx].tolist()
                map_f = lambda x: unique.index(x)
                unique_idx = list(map(map_f, sim_items_i))
                H[idx, unique_idx] = metrics[idx]

            for i, item in enumerate(seq_list):
                ego_idx = unique.index(item)
                H[i, ego_idx] = 1.0
                # multi-behavior hyperedge
                # if item in multibeh_group:
                #     H[i, n_unique+multibeh_group.index(item)] = 1.0
            # Add similarity edges for short sequences
            if len(seqs[batch_idx]) < self.max_sim_length:
                for i, item_i in enumerate(unique_items):
                    idx_i = torch.where(unique_items == item_i)[0]
                    for j, item_j in enumerate(unique_items):
                        idx_j = torch.where(unique_items == item_j)[0]
                        if item_sim[idx_i, idx_j] >= self.sim_threshold:
                            H[batch_idx, i, j] = item_sim[idx_i, idx_j]
            
            # print(H.detach().cpu().tolist())
            # W = torch.ones(n_edge, device=H.device)
            # W = torch.diag(W)
            DV = torch.sum(H, dim=1)
            DE = torch.sum(H, dim=0)
            invDE = torch.diag(torch.pow(DE, -1))
            invDV = torch.diag(torch.pow(DV, -1))
            # DV2 = torch.diag(torch.pow(DV, -0.5))
            HT = H.t()
            G = invDV.mm(H).mm(invDE).mm(HT)
            # G = DV2.mm(H).mm(invDE).mm(HT).mm(DV2)
            assert not torch.isnan(G).any()
            Gs.append(G.to(seqs.device))
        Gs_block_diag = torch.block_diag(*Gs)
        return Gs_block_diag


    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]                 [[ 0,  0,  0,  ..., 44, 47, 49],...[ 0,  0,  0,  ...,  0,  0,  1]]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1) #torch.Size([2560]) 
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot #torch.Size([2560, 200])

    def cal_loss(self, batch_seqs): #batch_seqs: list
        if self.dataset == "ml-20m" or self.dataset == "beauty": 
            session_id, item_seq, last_buy = batch_seqs
            item_type = None
        else:
            session_id, item_seq, item_type, last_buy = batch_seqs
        # item_seq = batch_seqs[self.ITEM_SEQ]
        # session_id = batch_seqs['session_id']
        # item_type = batch_seqs["item_type_list"]
        # last_buy = batch_seqs["item_id"]
        masked_item_seq, pos_items, masked_index, item_type_seq = self._transform_train_seq(item_seq, item_type, last_buy)
        mask_nums = torch.count_nonzero(pos_items, dim=1)
        seq_output = self.forward(masked_item_seq, item_type_seq, mask_positions_nums=(masked_index, mask_nums), session_id=session_id)

        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot #创建掩码位置的多热编码映射
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H] #除了mask位置其他都置0 #提取mask位置(purchase位置)的预测值
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        test_item_emb = self.item_embedding.weight  # [item_num H]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len] #mask 位置

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num] #计算预测值和所有item emb相似度
        # print(logits.size()) torch.Size([512, 40, 41533])
        # print(pos_items.view(-1).size()) torch.Size([20480])
        loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
            / torch.sum(targets)
        # pos_items ： 包含了掩码位置的真实物品ID [batch_size, mask_len]
        #损失只针对掩码位置进行计算，并且通过targets进行加权，其中targets表示掩码位置。
        #如果一个被掩码的位置的真实物品ID是 5，那么在计算损失时，这个位置的目标输出就是物品类别 5 的预测概率。
        # if self.enable_en:
        #     logits_2 = torch.matmul(seq_output_ed, test_item_emb.transpose(0, 1)) 
        #     loss_2 = torch.sum(loss_fct(logits_2.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
        #             / torch.sum(targets)
        #     total_loss = 0.8 * loss + 0.2 * loss_2
        #     # total_loss = loss_2
        # else:
        #     total_loss = loss
        loss_dict = {'rec_loss': loss.item()}
        
        return loss, loss_dict
        # return loss

    def full_predict(self, batch_seqs):
        if self.dataset == "ml-20m" or self.dataset == "beauty":
            session_id, item_seq, _ = batch_data
            type_seq = None
        else:
            session_id, item_seq, type_seq, _ = batch_data

        # item_seq = batch_seqs['item_id_list']
        # type_seq = batch_seqs['item_type_list']
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self._transform_test_seq(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(item_seq, type_seq)
        # if self.enable_en:
        #     seq_output = self.en_model(seq_output)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding.weight[:self.item_num]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores

    # def customized_sort_predict(self, batch_seqs):
    #     item_seq = batch_seqs['item_id_list']
    #     type_seq = batch_seqs['item_type_list']
        
    #     truth = batch_seqs['item_id']
    #     if self.dataset == "ijcai_beh":
    #         raw_candidates = [73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355]
    #     elif self.dataset == "retail_beh":
    #         raw_candidates = [101, 11, 14, 493, 163, 593, 1464, 12, 297, 123, 754, 790, 243, 250, 508, 673, 1161, 523, 41, 561, 2126, 196, 1499, 1093, 1138, 1197, 745, 1431, 682, 1567, 440, 1604, 145, 1109, 2146, 209, 2360, 426, 1756, 46, 1906, 520, 3956, 447, 1593, 1119, 894, 2561, 381, 939, 213, 1343, 733, 554, 2389, 1191, 1330, 1264, 2466, 2072, 1024, 2015, 739, 144, 1004, 314, 1868, 3276, 1184, 866, 1020, 2940, 5966, 3805, 221, 11333, 5081, 685, 87, 2458, 415, 669, 1336, 3419, 2758, 2300, 1681, 2876, 2612, 2405, 585, 702, 3876, 1416, 466, 7628, 572, 3385, 220, 772]
    #     elif self.dataset == "tmall_beh":
    #         raw_candidates = [2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701]
    #     customized_candidates = list()
    #     for batch_idx in range(item_seq.shape[0]):
    #         seen = item_seq[batch_idx].cpu().tolist()
    #         cands = raw_candidates.copy()
    #         for i in range(len(cands)):
    #             if cands[i] in seen:
    #                 new_cand = random.randint(1, self.item_num)
    #                 while new_cand in seen:
    #                     new_cand = random.randint(1, self.item_num)
    #                 cands[i] = new_cand
    #         cands.insert(0, truth[batch_idx].item()) 
    #         customized_candidates.append(cands)
    #     candidates = torch.LongTensor(customized_candidates).to(item_seq.device)
    #     item_seq_len = torch.count_nonzero(item_seq, 1)
    #     item_seq, type_seq = self._transform_test_seq(item_seq, item_seq_len, type_seq)
    #     seq_output = self.forward(item_seq, type_seq)
    #     if self.enable_en:
    #         seq_output = self.en_model(seq_output)
    #     seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
    #     test_items_emb = self.item_embedding(candidates)  # delete masked token
    #     scores = torch.bmm(test_items_emb, seq_output.unsqueeze(-1)).squeeze()  # [B, item_num]
    #     return scores

    # def sample_sequence(item_seq, item_type, max_length=20):
    #     sampled_item_seqs = []
    #     sampled_item_types = []

    #     for i in range(item_seq.size(0)):
    #         seq = item_seq[i]
    #         types = item_type[i]
    #         if len(seq) > max_length:
    #             prob = random.random()
    #             if prob < 0.5:
    #                 indices = sorted(random.sample(range(len(seq)), max_length))
    #                 seq_sampled = seq[indices]
    #                 types_sampled = types[indices]
    #         else:
    #             seq_sampled = seq
    #             types_sampled = types

    #         sampled_item_seqs.append(seq_sampled)
    #         sampled_item_types.append(types_sampled)

    #     sampled_item_seqs = torch.stack(sampled_item_seqs)
    #     sampled_item_types = torch.stack(sampled_item_types)

    #     return sampled_item_seqs, sampled_item_types



