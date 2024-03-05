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
        # self.loss_type = configs['model']['loss_type']
        self.hglen = configs['model']['hyper_len']
        self.enable_hg = configs['tune']['enable_hg']
        self.enable_ms = configs['tune']['enable_ms']
        self.dataset = configs['data']['name']
        self.max_seq_length = configs['model']['max_seq_len']
        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.hidden_size, self.max_seq_length)

        # self.buy_type = dataset.field2token_id["item_type_list"]['0']
        self.mask_token_lsy = self.item_num
        self.mask_token = self.item_num + 1
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
        # self.LayerNorm_lsy = nn.LayerNorm(self.hidden_size) #, eps=self.layer_norm_eps)
        # self.dropout = nn.Dropout(self.hidden_dropout_prob)

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
        # self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

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
        # try:
        #     assert self.loss_type in ['BPR', 'CE']
        # except AssertionError:
        #     raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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

    def _transform_test_seq(self, item_seq, item_seq_len):
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
        return item_seq

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-scale attention."""
        if self.enable_ms:
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1)
            
            # attention_mask = (item_seq > 0).unsqueeze(1).repeat(
            #     1, item_seq.size(1), 1).unsqueeze(1)
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

    def forward(self, batch_seqs):
        extended_attention_mask = self.get_attention_mask(batch_seqs) #torch.Size([512, 1, 200])
        # mask = (batch_seqs > 0).unsqueeze(1).repeat(
        #     1, batch_seqs.size(1), 1).unsqueeze(1)
        item_emb = self.emb_layer(batch_seqs) #torch.Size([512, 200, 64]) 
        trm_output = self.trm_encoder(item_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output

    def forward_ori(self, item_seq):
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        # input_emb = self.LayerNorm_lsy(input_emb)
        # x_raw = self.dropout(input_emb)  # torch.Size([128, 200, 64])
        # extended_attention_mask = self.get_attention_mask(item_seq)
        extended_attention_mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)
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
            pos = (item_seq[batch_idx]==self.mask_token_lsy).nonzero(as_tuple=True)[0][0]
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
            row_idx, masked_pos = torch.nonzero(sim_items==self.mask_token_lsy, as_tuple=True)
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
            normal_item_indexes = torch.nonzero((seq != self.mask_token_lsy), as_tuple=True)[0]
            for idx in normal_item_indexes:
                sim_items_i = sim_items[idx].tolist()
                map_f = lambda x: unique.index(x)
                unique_idx = list(map(map_f, sim_items_i))
                H[idx, unique_idx] = metrics[idx]

            # for i, item in enumerate(seq_list):
            #     ego_idx = unique.index(item)
            #     H[i, ego_idx] = 1.0
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
      
        return loss

    def full_predict(self, batch_data):
        if self.dataset == "ml-20m" or self.dataset == "beauty":
            batch_user, batch_seqs, batch_last_items = batch_data
            type_seq = None
        else:
            batch_user, batch_seqs, type_seq, batch_last_items = batch_data

        # item_seq = batch_seqs['item_id_list']
        # type_seq = batch_seqs['item_type_list']
        # item_seq_len = torch.count_nonzero(item_seq, 1)
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores = self.forward(masked_seqs)
        scores = scores[:, -1, :]
        return scores

    # def build_graph(self, item_seq):
    #     unique_items = torch.unique(item_seq[item_seq != self.mask_token])

    #     # Initialize adjacency matrix
    #     adj_matrix = torch.zeros((item_seq.size(0), len(unique_items), len(unique_items)))

    #     for batch_idx in range(item_seq.size(0)):
    #         # Add adjacent edges for items in the sequence
    #         for i in range(len(item_seq[batch_idx]) - 1):
    #             if item_seq[batch_idx, i] != self.mask_token and item_seq[batch_idx, i + 1] != self.mask_token:
    #                 idx_i = torch.where(unique_items == item_seq[batch_idx, i])[0]
    #                 idx_j = torch.where(unique_items == item_seq[batch_idx, i + 1])[0]
    #                 adj_matrix[batch_idx, idx_i, idx_j] = 1.0

    #         # Add similarity edges for short sequences
    #         if len(item_seq[batch_idx]) < self.max_sim_length:
    #             for i, item_i in enumerate(unique_items):
    #                 idx_i = torch.where(unique_items == item_i)[0]
    #                 for j, item_j in enumerate(unique_items):
    #                     idx_j = torch.where(unique_items == item_j)[0]
    #                     if self.sim_matrix[idx_i, idx_j] >= self.sim_threshold:
    #                         adj_matrix[batch_idx, i, j] = self.sim_matrix[idx_i, idx_j]
    #     return adj_matrix