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
        # self.enable_hg = configs['model']['enable_hg']
        # self.enable_ms = configs['model']['enable_ms']
        self.dataset = configs['data']['name']
        self.max_len = configs['model']['max_seq_len']
        self.dropout_rate = configs['model']['dropout_rate']
        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.hidden_size, self.max_len)

        self.emb_size = configs['model']['embedding_size']
        self.mask_token = self.item_num + 1
        self.transformer_layers_1 = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.emb_size * 4, self.dropout_rate) for _ in range(self.n_layers)])
        self.transformer_layers_2 = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.emb_size * 2, self.dropout_rate) for _ in range(self.n_layers)])
        self.transformer_layers_3 = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads * 2, self.emb_size * 8, self.dropout_rate) for _ in range(self.n_layers)])

        self.A_N = nn.Parameter(torch.Tensor(self.max_len, self.max_len))
        nn.init.xavier_uniform_(self.A_N.data)
        # self.reg_lambda = 0.001

        self.att_aggre = configs['model']['att_aggre']
        self.li_aggre_w = configs['model']['li_aggre_w']
        self.global_hg = configs['model']['global_hg']
        self.feed = configs['model']['feed']
        self.t = configs['model']['temperature']

        self.SelfAttentionModel = SelfAttentionModel(self.hidden_size, self.max_len, self.n_heads)
        self.MLP = MLP(self.max_len, self.max_len, self.max_len, num_layers=1) #hidden_layer =1, outputlayer = 1, total = 2

        self.A_I = nn.Parameter(torch.Tensor(self.item_num + 2, self.item_num + 2))
        nn.init.xavier_uniform_(self.A_I.data)
        # Apply prior to A_N (set diagonal elements to 0)
        self.A_N.data = self._apply_prior(self.A_N.data, self.max_len)
        self.A_I.data = self._apply_prior(self.A_I.data, self.item_num + 2)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps= 1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hgnn_layer = HGNN(self.hidden_size)

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

        self.hypergraphs = dict()

        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.out_fc = nn.Linear(self.hidden_size, self.item_num + 1)

        self.emb_nn = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),nn.Sigmoid())#, nn.Dropout(0.1))
        self.emb_nn[0].apply(lambda module: nn.init.uniform_(module.weight.data,0,0.001))
        self.emb_nn[0].apply(lambda module: nn.init.uniform_(module.bias.data,0,0.001))

        # self.global_nn = nn.Sequential(nn.Linear(self.max_len, self.max_len),nn.Sigmoid())#, nn.Dropout(0.1))
        # self.global_nn[0].apply(lambda module: nn.init.uniform_(module.weight.data,0,0.001))
        # self.global_nn[0].apply(lambda module: nn.init.uniform_(module.bias.data,0,0.001))

        self.global_map = nn.Parameter(torch.Tensor(self.max_len, self.max_len))
        nn.init.normal_(self.global_map, std=0.02)
        self.global_map_bias = nn.Parameter(torch.Tensor(1, self.max_len))
        nn.init.normal_(self.global_map_bias, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _apply_prior(self, matrix, item_count):
        # Set diagonal elements to 0
        matrix[torch.arange(item_count), torch.arange(item_count)] = 0
        return matrix

    def reg_loss(self):
        # L2 regularization on the A_n matrix
        return torch.norm(self.A_N, p=2)

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
        mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)
        x = self.emb_layer(item_seq) #b, max_l, h 
        seq_contribution_score = self.SelfAttentionModel(x)#.unsqueeze(-1).expand(-1, -1, x.size(-1)) # B,L -> b,l,l
        # 不直接计算b,l,h， 让h维度的的importance共享！！
        global_contribution_score = self.MLP(seq_contribution_score) # B,L
        x_1_weight = seq_contribution_score.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
        x_2_weight = global_contribution_score.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
        for transformer in self.transformer_layers_1:
            x_1 = transformer(torch.matmul(x_1_weight, x), mask) # [b, l]
        for transformer in self.transformer_layers_2:
            x_2 = transformer(torch.matmul(x_2_weight, x), mask) # [b, l]
        mixed_x = x_1 + x_2
        #3_13
        # x_l = self.emb_nn(x) # project into compatatbility space, sigmoid score: 0-1 # b, l ,h 但是embedding本来就在一个空间
        # comp_matrix = self.cos_sim(x)#(x_l) ##similarity matrix #compatible score # b, l ,l #cos:[-1,1]
        # comp_matrix_nor = F.normalize(comp_matrix, p=2, dim=(1, 2))
        # global_comp = comp_matrix_nor.matmul(self.global_map)+self.global_map_bias # b, l ,l 但是值域和local不一样
        # global_comp_nor = F.normalize(global_comp, p=2, dim=(1, 2)) #normalize 让两个matrix不要相差太多
        # overall_comp = comp_matrix_nor + global_comp_nor # b, l, l
        # new_x = torch.matmul(torch.sigmoid(overall_comp), x)
        # for transformer in self.transformer_layers_1:
        #     x_1 = transformer(new_x, mask)
        # for transformer in self.transformer_layers_2:
        #     x_2 = transformer(x, mask)
        # mixed_x = x_1 + x_2
        # 3_12
        # seq_contribution_score = torch.sigmoid(torch.sum(torch.matmul(compatibility_score_matrix, x)), dim=-1) 
        # #b,l,l * b,l,h-> b,l,h 
        # print(seq_contribution_score.size())
        # global_contribution_score = self.global_nn(seq_contribution_score) #sigmoid(linear) #b, l
        # total_contribution_scores = seq_contribution_score + global_contribution_score # b, l
        # # Normalize total contribution scores
        # total_contribution_scores = F.normalize(total_contribution_scores, p=1, dim=1)
        # # Element-wise multiplication with input sequence
        # denoised_sequence = x * total_contribution_scores.unsqueeze(2)
        # # Sum along sequence length dimension
        # new_x = torch.sum(denoised_sequence, dim=1)
        # for transformer in self.transformer_layers_1:
        #     mixed_x = transformer(new_x, mask)

        # 3_13
        # x_raw = x * torch.sigmoid(x.matmul(self.gating_weight)+self.gating_bias) # b, l, h #func. 12?
        # # b, l, l #passed through a gating mechanism involving a sigmoid function
        # x_m = torch.stack((self.metric_w1*x_raw, self.metric_w2*x_raw)).mean(0)  # b, l, h
        # item_sim = self.sim(x_m, x_m) #similarity matrix #compatible score # b, l ,l
        # # item_sim[item_sim < 0] = 0.01 # 不加才是去噪
        # hg = self.hgnn_layer(x_m, item_sim)

        # for transformer in self.transformer_layers_1:
        #     x_1 = transformer(x_m, mask)
        # # mixed_x = self.li_aggre_w * x_1 + (1-self.li_aggre_w) * hg
        # mixed_x = x_1 + hg
        # 3_11
        # if self.global_hg:
        #     A_K = []
        #     for batch_idx in range(item_seq.shape[0]):
        #         seq = item_seq[batch_idx]
        #         # Ensure that seq contains valid indices
        #         if (seq < 0).any() or (seq >= self.A_I.size(0)).any():
        #             raise ValueError("Invalid indices in seq.")
        #         A_I_seq = self.A_I.index_select(0, seq).index_select(1, seq) # 50,50
        #         assert not torch.isnan(A_I_seq).any()
        #         A_K_seq = A_I_seq + self.A_N #0.5 * ((A_I_seq + self.A_N) + (A_I_seq + self.A_N).t())
        #         A_K.append(A_K_seq.unsqueeze(0))  # Add a batch dimension

        #     A_K = torch.cat(A_K, dim=0) #torch.Size([512, 50, 2500])
        # else: 
        #     A_K = self.A_N.unsqueeze(0).expand(x.size(0), -1, -1)

        # x_k = torch.matmul(A_K, x)

        # if self.feed:
        #     for transformer in self.transformer_layers_1:
        #         mixed_x = transformer(x_k, mask)

        # else:
        #     for transformer in self.transformer_layers_1:
        #         x_1 = transformer(x, mask)

        #     if self.att_aggre: 
        #         mixed_x = torch.stack((x_1, x_k), dim=0)
        #         weights = (torch.matmul(mixed_x, self.attn_weights.unsqueeze(0).unsqueeze(0))*self.attn).sum(-1)
        #         score = F.softmax(weights, dim=0).unsqueeze(-1)
        #         mixed_x = (mixed_x*score).sum(0)
        #     elif self.li_aggre:
        #         mixed_x = self.li_aggre_w * x_1 + (1-self.li_aggre_w) * x_k

        # 3_3
        # for transformer in self.transformer_layers_2:
        #     x_2 = transformer(hgnn_embs, mask)
        # for transformer in self.transformer_layers_3:
        #     x_3 = transformer(hgnn_embs, mask)
        return mixed_x #, x_1, x_2 # + x_2 + x_3

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        masked_seqs, masked_items = self._transform_train_seq(
            batch_seqs, batch_last_items.unsqueeze(1))
        # B, T, E
        logits_t = self.forward(masked_seqs) # [b, l]
        # loss_contrstive = torch.mean((logits_t - logits_g)**2)
        logits = self.out_fc(logits_t) # [b, l, n+1]
        # B, T, E -> B*T, E
        logits = logits.view(-1, logits.size(-1)) # [b*l, n+1]
        # kl_loss = F.kl_div(logits_1, logits_2, reduction='batchmean')
        loss = self.loss_func(logits, masked_items.reshape(-1)) 
        total_loss = loss #+ 0.1 * kl_loss 
        loss_dict = {'rec_loss': total_loss.item()}#, "kl_loss": kl_loss.item()}#, "reg_loss": reg_loss.item()}
        return total_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores = self.forward(masked_seqs)
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

    def cos_sim(self, x):
        # Compute dot product of all pairs of vectors
        dot_product = torch.matmul(x, x.transpose(1, 2))
        magnitudes = torch.norm(x, p=2, dim=-1, keepdim=True)
        similarity_matrix = dot_product / (magnitudes * magnitudes.transpose(1, 2) + 1e-8)
        return similarity_matrix

    def distillation_loss(self, outputs, teacher_outputs, temperature):
        soft_teacher_outputs = F.softmax(teacher_outputs / temperature, dim=1)
        # 计算学生模型的 softmax 概率分布
        soft_outputs = F.softmax(outputs / temperature, dim=1)
        loss = -torch.mean(torch.sum(soft_teacher_outputs * torch.log(soft_outputs), dim=1))
        return loss

    def kl_loss(self, outputs, teacher_outputs, temperature):
        soft_teacher_outputs = F.log_softmax(teacher_outputs / temperature, dim=1)
        soft_outputs = F.log_softmax(outputs / temperature, dim=1)
        loss = F.kl_div(soft_outputs, soft_teacher_outputs, reduction='batchmean')
        return loss