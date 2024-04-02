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
        # self.hglen = configs['model']['hyper_len']
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

        self.feed = configs['model']['feed']
        self.batch_size = configs['train']['batch_size']
        self.lmd = configs['model']['lmd']
        self.tau = configs['model']['tau']
        self.cl_weight = configs['model']['cl_weight']
        self.replace_ratio = configs['model']['replace_ratio']
        self.sim_method = configs['model']['sim_method']
        self.with_contri = configs['model']['with_contri']
        self.cont_method = configs['model']['cont_method']
        
        self.SelfAttentionModel = SelfAttentionModel(self.hidden_size, self.max_len, self.n_heads)
        self.SelfAttentionModel_2 = SelfAttentionModel(self.hidden_size, self.max_len, self.n_heads)
        self.MLP = MLP(self.max_len, self.max_len, self.max_len, num_layers=1) #hidden_layer =1, outputlayer = 1, total = 2
        # self.l_MLP = MLP(self.max_len * self.hidden_size, self.max_len, self.max_len, num_layers=1)
        # self.multi_head = MultiHeadAttention(num_heads=self.n_heads, hidden_size=self.hidden_size, dropout=self.dropout_rate)
        self.mask_default = self.mask_correlated_samples(
            batch_size= self.batch_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps= 1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.gating_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        nn.init.normal_(self.gating_bias, std=0.02)
        nn.init.normal_(self.gating_weight, std=0.02)
        nn.init.normal_(self.metric_w1, std=0.02)
        nn.init.normal_(self.metric_w2, std=0.02)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.out_fc = nn.Linear(self.hidden_size, self.item_num + 1)
        self.cl_loss_func = nn.CrossEntropyLoss()
        if self.dataset == "sports":
            self.emb_nn = nn.Sequential(nn.Linear(self.hidden_size, 8),nn.Sigmoid())
        else:
            self.emb_nn = nn.Sequential(nn.Linear(self.hidden_size, 16),nn.Sigmoid())#, nn.Dropout(0.1))
        self.emb_nn[0].apply(lambda module: nn.init.uniform_(module.weight.data,0,0.001))
        self.emb_nn[0].apply(lambda module: nn.init.uniform_(module.bias.data,0,0.001))

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

    # 基于邻近项的补充
    def impute_missing_items(self, input_seq, scores, topk =1): # inputseq: not padded yet
        item_candidates = self.emb_layer.token_emb.weight[:, :]
        input_seq_1 = input_seq.clone().detach()
        input_seq_2 = input_seq.clone().detach()
        # if self.candi == "all":
        #     item_candidates = self.emb_layer.token_emb.weight[:, :]#.weight.clone().detach()
        # else:
        #     item_candidates = self.emb_layer.token_emb(input_seq)
        # print(embedding_weights.requires_grad) #True
        embedding_weights = self.emb_nn(item_candidates)
        new_score = scores.clone().detach()
        # print(new_score.requires_grad) #False
        # similarity_matrix = F.cosine_similarity(embedding_weights.unsqueeze(1), embedding_weights.unsqueeze(0), dim=-1) #太占内存！！
        if self.sim_method == "mm":
            similarity_matrix = torch.mm(embedding_weights, embedding_weights.T).to(input_seq.device)
        elif self.sim_method == "cos":
            similarity_matrix = self.sim(embedding_weights, embedding_weights).to(input_seq.device)
        elif self.sim_method == "gate":
            embedding_weights = embedding_weights * torch.sigmoid(embedding_weights.matmul(self.gating_weight)+self.gating_bias)
            # b, l, l
            x_m = torch.stack((self.metric_w1*embedding_weights, self.metric_w2*embedding_weights)).mean(0)
            similarity_matrix = self.sim(x_m, x_m).to(input_seq.device)
            # item_sim[item_sim < 0] = 0.01

        # 获取对角线索引
        diag_indices = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        # 将对角线元素设置为0
        similarity_matrix[diag_indices, diag_indices] = 0 #[15418, 15418] item_num+2
        # Get the item with highest score
        max_contribution, top1_indices = scores.topk(k=1, dim=-1)  # (batch_size, 1) # contribution 最高的item
        min_contribution, min_index = scores.topk(k=1, dim=-1, largest = False)
        for i, seq in enumerate(input_seq_1): #i:batch_index     
            prob = random.random()
            if prob < self.replace_ratio:
                top1_item_idx = top1_indices[i, 0]
                item_id = input_seq_1[i, top1_item_idx] #5062, 6275 #top1_indices[i, 0]: 每一个batch第一个
                similar_items = similarity_matrix[item_id.long()].topk(k=1)[1] #10585: index
                input_seq_2[i, min_index[i, 0]] = similar_items
                input_seq_1[i, min_index[i, 0]] = item_id
                new_score[i, min_index[i, 0]] = max_contribution[i, 0]
        return input_seq_1, input_seq_2, new_score

    def forward(self, item_seq):
        mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)
        x = self.emb_layer(item_seq) #b, max_l, h 
        if self.cont_method == "att":
            seq_contribution_score = self.SelfAttentionModel(x)
        else:
            seq_contribution_score = self.MLP(x)
        # 不直接计算b,l,h， 让h维度的的importance共享！！
        new_item_seq_1, new_item_seq_2, pad_scores = self.impute_missing_items(item_seq, seq_contribution_score)
        new_x_emb_1 = self.emb_layer(new_item_seq_1)
        new_mask_1 = (new_item_seq_1 > 0).unsqueeze(1).repeat(1, new_item_seq_1.size(1), 1).unsqueeze(1)
        new_x_emb_2 = self.emb_layer(new_item_seq_2)
        new_mask_2 = (new_item_seq_2 > 0).unsqueeze(1).repeat(1, new_item_seq_2.size(1), 1).unsqueeze(1)
        
        if self.with_contri:
            # new_x, pad_scores = x, seq_contribution_score
            # global_contribution_score = self.MLP(pad_scores) # B,L
            global_contribution_score = self.SelfAttentionModel_2(new_x_emb_2)
            x_1_weight = pad_scores.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
            # x_1_weight = seq_contribution_score.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
            x_2_weight = global_contribution_score.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
            for transformer in self.transformer_layers_1:
                x_1 = transformer(torch.matmul(x_1_weight, new_x_emb_1), new_mask_1) # [b, l, h]: LX
            for transformer in self.transformer_layers_1:
                x_2 = transformer(x, mask) # [b, l, h] : X
            for transformer in self.transformer_layers_1:
                x_3 = transformer(torch.matmul(x_2_weight, new_x_emb_2), new_mask_2) # [b, l, h] :GX
        else:
            for transformer in self.transformer_layers_1:
                x_1 = transformer(new_x_emb_1, new_mask_1) # [b, l, h]: LX
            for transformer in self.transformer_layers_1:
                x_2 = transformer(x, mask) # [b, l, h] : X
            for transformer in self.transformer_layers_1:
                x_3 = transformer(new_x_emb_2, new_mask_2) # [b, l, h] :GX
        mixed_x = x_1 + x_2 + x_3
        return mixed_x, x_1, x_2, x_3 # LX, X, GX
        
    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        masked_seqs, masked_items = self._transform_train_seq(
            batch_seqs, batch_last_items.unsqueeze(1))
        # masked_items #torch.Size([b, l]):tensor([[   0,    0,    0,  ...,    0,    0, 7647],...]])
        # print(masked_items.reshape(-1).size()) # b*l
        logits_t, LX, X, GX = self.forward(masked_seqs) # ([b, l, h])
        # logits = self.out_fc(logits_t) # [b, l, n+1]
        logits = self.out_fc(X) # [b, l, n+1]
        # LX = self.out_fc(LX)
        # GX = self.out_fc(GX) # 实验证明不太好用！！
        LX = LX[:, -1, :] #最后一个item B,H
        GX = GX[:, -1, :]
        logits = logits.view(-1, logits.size(-1)) # [b*l, n+1]
        #logits:tensor([[-0.0051,  0.0019, -0.0078,  ..., -0.0057, -0.0004, -0.0040],...]])
        loss = self.loss_func(logits, masked_items.reshape(-1)) 
        cl_loss = self.lmd * self.info_nce(
            LX, GX, temp=self.tau, batch_size=LX.shape[0])

        loss_dict = {
            'rec_loss': loss.item(),
            'cl_loss': cl_loss.item(),
        }
        return loss + self.cl_weight * cl_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores, LX, X, GX = self.forward(masked_seqs)
        # scores = self.out_fc(scores) # [b, l, n+1]
        scores = self.out_fc(X)
        scores = scores[:, -1, :] #提取每个样本序列中最后一个位置的分数
        return scores

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # return torch.matmul(z1, z2.permute(0,2,1))
        return torch.matmul(z1, z2.permute(1,0))

    def cos_sim(self, x):
        # Compute dot product of all pairs of vectors
        dot_product = torch.matmul(x, x.transpose(1, 2))
        magnitudes = torch.norm(x, p=2, dim=-1, keepdim=True)
        similarity_matrix = dot_product / (magnitudes * magnitudes.transpose(1, 2) + 1e-8)
        return similarity_matrix

    def info_nce(self, z_i, z_j, temp, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)
        return info_nce_loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

