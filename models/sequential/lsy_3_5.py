import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.model_utils import *
from models.base_model import BaseModel
from config.configurator import configs

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
        self.mask_prob = configs['model']['mask_ratio']
        self.loss_type = configs['model']['loss_type']
        self.hglen = configs['model']['hyper_len']
        self.enable_hg = configs['tune']['enable_hg']
        self.enable_ms = configs['tune']['enable_ms']
        self.dataset = configs['data']['name']
        self.max_len = configs['model']['max_seq_len']
        self.dropout_rate = configs['model']['dropout_rate']
        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.hidden_size, self.max_len)
        # self.layer_configs = [[64, 2], [256, 4]]
        # self.transformer_layers = MultiScaleTransformer(self.layer_configs, feed_forward_size=self.hidden_size * 4, dropout_rate=self.dropout_rate)
            # self.hidden_size, self.n_heads, self.hidden_size * 4, self.dropout_rate, scales=configs['model']["scales"])
        # self.transformer_layers = MultiScaleTransformerLayer(
        #     self.hidden_size, self.n_heads, self.hidden_size * 4, self.dropout_rate, scale_factors=configs['model']["scales"])

        # self.buy_type = dataset.field2token_id["item_type_list"]['0']
        self.emb_size = configs['model']['embedding_size']
        self.mask_token = self.item_num + 1
        self.transformer_layers_1 = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.emb_size * 4, self.dropout_rate) for _ in range(self.n_layers)])
        self.transformer_layers_2 = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.emb_size * 2, self.dropout_rate) for _ in range(self.n_layers)])
        self.transformer_layers_3 = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads * 2, self.emb_size * 8, self.dropout_rate) for _ in range(self.n_layers)])
        # self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        # self.type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        # self.item_embedding = nn.Embedding(self.item_num +1, self.hidden_size, padding_idx=0)  # mask token add 1
        # self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)  # add mask_token at the last

        # if self.enable_ms:
        #     self.trm_encoder = TransformerEncoder(
        #         n_layers=self.n_layers,
        #         num_heads=self.n_heads,
        #         hidden_size=self.hidden_size,
        #         inner_size=self.inner_size,
        #         hidden_dropout_prob=self.hidden_dropout_prob,
        #         attn_dropout_prob=self.attn_dropout_prob,
        #         hidden_act=self.hidden_act,
        #         layer_norm_eps=self.layer_norm_eps,
        #         multiscale=True,
        #         scales=configs['model']["scales"]
        #     )
        # else:
        #     self.trm_encoder = TransformerEncoder(
        #         n_layers=self.n_layers,
        #         num_heads=self.n_heads,
        #         hidden_size=self.hidden_size,
        #         inner_size=self.inner_size,
        #         hidden_dropout_prob=self.hidden_dropout_prob,
        #         attn_dropout_prob=self.attn_dropout_prob,
        #         hidden_act=self.hidden_act,
        #         layer_norm_eps=self.layer_norm_eps,
        #         multiscale=False
        #     )

        self.hgnn_layer = HGNN(self.hidden_size)
        # self.LayerNorm = nn.LayerNorm(self.hidden_size) #, eps=self.layer_norm_eps)
        # self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # self.hg_type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.gating_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # self.position_range = 3
        self.sim_threshold = 0.1
        self.max_sim_length = 6
        
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        nn.init.normal_(self.gating_bias, std=0.02)
        nn.init.normal_(self.gating_weight, std=0.02)
        nn.init.normal_(self.metric_w1, std=0.02)
        nn.init.normal_(self.metric_w2, std=0.02)

        # if self.dataset == "retail_beh":
        #     self.sw_before = 10
        #     self.sw_follow = 6
        # elif self.dataset == "ijcai_beh":
        #     self.sw_before = 30
        #     self.sw_follow = 18
        # elif self.dataset == "tmall_beh":
        #     self.sw_before = 20
        #     self.sw_follow = 12
        # elif self.dataset == "ml-20m":
        #     self.sw_before = 4
        #     self.sw_follow = 2
        # elif self.dataset == "beauty":
        #     self.sw_before = 4
        #     self.sw_follow = 2

        self.hypergraphs = dict()

        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.out_fc = nn.Linear(self.hidden_size, self.item_num + 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _transform_train_seq(self, batch_seqs, batch_last_items):
        device = batch_seqs.device
        seqs = torch.cat([batch_seqs, batch_last_items], dim=1)
        seqs = seqs.tolist()
        masked_seqs = []
        masked_items = []
        for seq in seqs: # for each seq
            masked_seq = []
            masked_item = []
            for item in seq: # for each item
                if item == 0: # ignore 0 idx (padding)
                    masked_seq.append(0)
                    masked_item.append(0)
                    continue
                prob = random.random()
                if prob < self.mask_prob: # mask
                    prob /= self.mask_prob
                    if prob < 0.8:
                        masked_seq.append(self.mask_token)
                    elif prob < 0.9: # replace
                        masked_seq.append(random.randint(1, self.item_num)) # both include
                    else: # keep
                        masked_seq.append(item)
                    masked_item.append(item)
                else: # not mask
                    masked_seq.append(item) # keep
                    masked_item.append(0) # 0 represent no item
            masked_seqs.append(masked_seq)
            masked_items.append(masked_item)
        masked_seqs = torch.tensor(masked_seqs, device=device, dtype=torch.long)[:, -self.max_len:]
        masked_items = torch.tensor(masked_items, device=device, dtype=torch.long)[:, -self.max_len:]
        return masked_seqs, masked_items

    def _transform_test_seq(self, batch_seqs):
        batch_mask_token = torch.LongTensor(
            [self.mask_token] * batch_seqs.size(0)).unsqueeze(1).to(batch_seqs.device)
        seqs = torch.cat([batch_seqs, batch_mask_token], dim=1)
        return seqs[:, -self.max_len:]

    def forward(self, item_seq):
        mask = (item_seq > 0).unsqueeze(1).repeat(
            1, item_seq.size(1), 1).unsqueeze(1)
        x = self.emb_layer(item_seq) #torch.Size([512, 50, 64])  
        x_raw = x * torch.sigmoid(x.matmul(self.gating_weight)+self.gating_bias) # torch.Size([128, 200, 64]) #func. 12?
        # b, l, l #passed through a gating mechanism involving a sigmoid function
        x_m = torch.stack((self.metric_w1*x_raw, self.metric_w2*x_raw)).mean(0) # torch.Size([128, 200, 64])
        item_sim = self.sim(x_m, x_m)
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
            # pos = (item_seq[batch_idx]==self.mask_token).nonzero(as_tuple=True)[0][0]
            # sliding_window_start = pos-self.sw_before if pos-self.sw_before>-1 else 0
            # embs[pos] = torch.mean(embs[sliding_window_start:pos], dim=0)
            hgnn_embs_padded.append(embs)
        # b, l, dim
        hgnn_embs = torch.stack(hgnn_embs_padded, dim=0)
        for transformer in self.transformer_layers_1:
            x_1 = transformer(x, mask)
        # for transformer in self.transformer_layers_2:
        #     x_2 = transformer(hgnn_embs, mask)
        # for transformer in self.transformer_layers_3:
        #     x_3 = transformer(hgnn_embs, mask)
        return x_1, hgnn_embs # + x_2 + x_3

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        masked_seqs, masked_items = self._transform_train_seq(
            batch_seqs, batch_last_items.unsqueeze(1))
        # B, T, E
        logits_t, logits_g = self.forward(masked_seqs) # [b, l]
        # loss_contrstive = torch.mean((logits_t - logits_g)**2)

        logits = self.out_fc(logits_t) # [b, l, n+1]
        # B, T, E -> B*T, E
        logits = logits.view(-1, logits.size(-1)) # [b*l, n+1]
        loss = self.loss_func(logits, masked_items.reshape(-1)) 

        logits_g = self.out_fc(logits_g) # [b, l, n+1]
        # B, T, E -> B*T, E
        logits_g = logits_g.view(-1, logits_g.size(-1)) # [b*l, n+1]
        loss_g = self.loss_func(logits_g, masked_items.reshape(-1)) 

        total_loss = loss * 0.8 + 0.2 * loss_g      
        loss_dict = {'rec_loss': total_loss.item(), 'loss_g': loss_g.item()}
        return total_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores, scores_hg = self.forward(masked_seqs)
        scores = self.out_fc(scores)
        scores = scores[:, -1, :]
        return scores

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
            n_edge = n_unique
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
    
            # # Add similarity edges for short sequences
            # if len(seqs[batch_idx]) < self.max_sim_length:
            #     for i, item_i in enumerate(unique_items):
            #         idx_i = torch.where(unique_items == item_i)[0]
            #         for j, item_j in enumerate(unique_items):
            #             idx_j = torch.where(unique_items == item_j)[0]
            #             if item_sim[idx_i, idx_j] >= self.sim_threshold:
            #                 H[batch_idx, i, j] = item_sim[idx_i, idx_j]
            
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

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.matmul(z1, z2.permute(0,2,1))