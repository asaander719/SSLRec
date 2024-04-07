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
        self.aug_ratio = configs['model']['aug_ratio']
        self.sim_method = configs['model']['sim_method']
        self.with_contri = configs['model']['with_contri']
        self.cont_method = configs['model']['cont_method']
        self.aug_k = configs['model']['aug_k']
        self.aug_with_maxself = configs['model']['aug_with_maxself'] #False
        self.aug_select = AugmentationSelector(self.max_len)
        if self.dataset == "sports":
            self.candidates = torch.tensor([7647, 13170, 441, 8887, 1502, 10933, 3652, 3012, 12372, 4122, 9750, 10090, 6266, 6275, 3533, 9794, 5961, 7203, 12874, 8969, 
                                             14066, 6118, 12911, 6569, 8763, 1930, 14204, 5265, 10075, 11145, 12691, 5863, 8635, 3127, 6616, 11306, 5633, 5858, 8213, 1587, 
                                             5626, 4806, 2221, 1535, 766, 592, 3505, 14675, 8147, 5958, 2259, 4666, 4263, 14056, 9232, 14199, 8602, 12779, 1194, 8261, 8250, 
                                             13354, 9262, 9381, 1226, 12918, 10062, 9298, 11899, 5438, 11, 9168, 13046, 8291, 2982, 6376, 12245, 1652, 11229, 11712, 940, 349, 
                                             2141, 418, 3488, 5566, 770, 7687, 12211, 10383, 11859, 7854, 1630, 3699, 11128, 9292, 5719, 14320, 9183, 1931, 8861, 1557, 15046, 
                                             8684, 3100, 3209, 8931, 11106, 12597, 6537, 5897, 11029, 9920, 8403, 8711, 7075, 12067, 10178, 10060, 14486, 2841, 14183, 13020, 
                                             15232, 8089, 3760, 4523, 1852, 1275, 14653, 14952, 7234, 14649, 2483, 14690, 6921, 6107, 598, 13589, 13680, 14483, 12504, 6691, 
                                             7154, 2778, 931, 7697, 8240, 4130, 1424, 7850, 7146, 964, 13681, 12778, 12689, 9059, 10827, 10305, 8669, 7641, 8838, 10889, 3246, 
                                             10101, 4241, 142, 14965, 4250, 2704, 10688, 13596, 12215, 2660, 3952, 14575, 14989, 6772, 13462, 12757, 7374, 3382, 4637, 5555, 
                                             2149, 13017, 818, 2535, 6394, 10415, 14581, 10877, 9017, 7750, 9810, 10036, 2607, 10896, 726, 14095])
        
        self.MLP = MLP(self.hidden_size*self.max_len, self.max_len, self.max_len, num_layers=1) #hidden_layer =1, outputlayer = 1, total = 2
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
            self.emb_nn = nn.Sequential(nn.Linear(self.hidden_size, 64),nn.Sigmoid())
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
    def impute_missing_items(self, input_seq, scores, p_aug, p_repleace): # inputseq:padded b,l # item_seq, seq_contribution_score, p_aug, p_repleace
        pad_token = 0  # 填充的标记
        scores[input_seq == pad_token] = 0 #float('-inf')
        # mask = input_seq != pad_token
        # scores[~mask] = float('-inf')
        # item_candidates = torch.unique(input_seq.flatten())
        # embedding_weights = self.emb_layer.token_emb(item_candidates)
        if self.dataset == "sports":
            input_seq_flatten = input_seq.flatten()
            item_candidates = torch.unique(torch.cat((input_seq_flatten, self.candidates.to(input_seq.device))))
            item_candidates = item_candidates[item_candidates !=0]    
        else:
            item_candidates = torch.arange(1, self.item_num+2).to(input_seq.device)
            # embedding_weights = self.emb_layer.token_emb.weight[:, :]
        # print(item_candidates)
        embedding_weights = self.emb_layer.token_emb(item_candidates)
        input_seq_1 = input_seq.clone().detach()
        input_seq_2 = input_seq.clone().detach()
        # embedding_weights = self.emb_nn(embedding_weights)
        # new_score = scores.clone().detach() # print(new_score.requires_grad) #False
        # print(scores)
        if self.sim_method == "mm":
            similarity_matrix = torch.mm(embedding_weights, embedding_weights.T).to(input_seq.device)
        elif self.sim_method == "cos":
            similarity_matrix = self.sim(embedding_weights, embedding_weights).to(input_seq.device)
        elif self.sim_method == "gate":
            embedding_weights = embedding_weights * torch.sigmoid(embedding_weights.matmul(self.gating_weight)+self.gating_bias) # b, l, l
            x_m = torch.stack((self.metric_w1*embedding_weights, self.metric_w2*embedding_weights)).mean(0)
            similarity_matrix = self.sim(x_m, x_m).to(input_seq.device)
            # item_sim[item_sim < 0] = 0.01

        if not self.aug_with_maxself: # 将对角线元素设置为0，用得分最高的item自己代替得分最小的
            diag_indices = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device) # 获取对角线索引
            similarity_matrix[diag_indices, diag_indices] = 0 #[15418, 15418] item_num+2
        
        max_contribution, max_indices = scores.topk(k=self.aug_k, dim=-1)  # (batch_size, k) # contribution 最高的item # Get the item with highest score
        min_contribution, min_index = scores.topk(k=self.aug_k, dim=-1, largest = False)
        # print(min_contribution)
        for i, seq in enumerate(input_seq_1): #i:batch_index 
            if (1-p_repleace[i]) * p_aug[i] > self.aug_ratio:    # 结合长度和分布共同决定是否进行变换
                if p_repleace[i] > self.replace_ratio:
                    for j in range(self.aug_k):
                        topk_item_idx = max_indices[i, j] #49
                        item_id = input_seq_1[i, topk_item_idx] #5062, 6275 #top1_indices[i, 0]: 每一个batch第一个
                        item_idx = torch.where(item_candidates == item_id)[0]
                        similar_items_idx = similarity_matrix[item_idx.long()].topk(k=1)[1] #10585: index
                        similar_items = item_candidates[similar_items_idx]
                        # print(topk_item_idx, item_id, item_idx, similar_items_idx, similar_items, min_index[i, j])
                        # 15, 0 (padding item), [], [], 17
                        input_seq_2[i, min_index[i, j]] = similar_items
                        # input_seq_1[i, min_index[i, j]] = item_id
                        # new_score[i, min_index[i, j]] = max_contribution[i, j]
                         # if prob < self.replace_ratio:
            #             top1_item_idx = top1_indices[i, 0]
            #             item_id = input_seq_1[i, top1_item_idx] #5062, 6275 #top1_indices[i, 0]: 每一个batch第一个
            #             similar_items = similarity_matrix[item_id.long()].topk(k=1)[1] #10585: index
            #             input_seq_2[i, min_index[i, 0]] = similar_items
            #             input_seq_1[i, min_index[i, 0]] = item_id
            #             new_score[i, min_index[i, 0]] = max_contribution[i, 0]
                else:
                    for j in range(self.aug_k):
                        topk_item_idx = max_indices[i, j]
                        item_id = input_seq_1[i, topk_item_idx] #5062, 6275 #top1_indices[i, 0]: 每一个batch第一个
                        item_idx = torch.where(item_candidates == item_id)[0]
                        similar_items_idx = similarity_matrix[item_idx.long()].topk(k=1)[1] #10585: index
                        similar_items = item_candidates[similar_items_idx]
                        position = torch.where(seq == item_id)[0].item() # Find the position of the item in input_seq_1
                        # Insert the similar item into input_seq_2
                        input_seq_2[i] = torch.cat((input_seq_2[i][:position], similar_items, input_seq_2[i][position:]), dim=0)           
            # print(input_seq, input_seq_1, input_seq_2)
        return input_seq_1, input_seq_2
    

    def forward(self, item_seq):
        mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)
        x = self.emb_layer(item_seq) #b, max_l, h 
        p_repleace = self.aug_select(item_seq) #b
        # print(x)
        # if self.cont_method == "att":
        #     seq_contribution_score = self.SelfAttentionModel(x)
        # else: #mlp
        seq_contribution_score = self.MLP(x) # b, l
        std_dev = torch.std(seq_contribution_score, dim=1) #标准差 b,l
        p_d= 1 - torch.exp(-std_dev)
        # Normalize probability to ensure it is between 0 and 1
        p_aug = (p_d - torch.min(p_d)) / (torch.max(p_d) - torch.min(p_d))  # b # [0.3093, 0.0935, 0.4187, 0.5053, 0.2575, 0.2001, 0.3147, 0.2539, 0.4102]
        new_item_seq_1, new_item_seq_2 = self.impute_missing_items(item_seq, seq_contribution_score, p_aug, p_repleace)
        new_x_emb_1 = self.emb_layer(new_item_seq_1)
        new_mask_1 = (new_item_seq_1 > 0).unsqueeze(1).repeat(1, new_item_seq_1.size(1), 1).unsqueeze(1)
        new_x_emb_2 = self.emb_layer(new_item_seq_2)
        new_mask_2 = (new_item_seq_2 > 0).unsqueeze(1).repeat(1, new_item_seq_2.size(1), 1).unsqueeze(1)

        if self.with_contri:
            # new_x, pad_scores = x, seq_contribution_score
            pad_scores = self.MLP(new_x_emb_1)
            global_contribution_score = self.MLP(new_x_emb_2) # B,L
            x_1_weight = pad_scores.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
            # x_1_weight = seq_contribution_score.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
            x_2_weight = global_contribution_score.unsqueeze(-1).expand(-1, -1, x.size(1)) # b,l,l
            for transformer in self.transformer_layers_1:
                # x_1 = transformer(torch.matmul(x_1_weight, new_x_emb_1), new_mask_1) # [b, l, h]: LX
                x_1 = transformer(new_x_emb_1, new_mask_1)
            for transformer in self.transformer_layers_1:
                x_2 = transformer(x, mask) # [b, l, h] : X
            for transformer in self.transformer_layers_1:
                # x_3 = transformer(torch.matmul(x_2_weight, new_x_emb_2), new_mask_2) # [b, l, h] :GX
                x_3 = transformer(new_x_emb_2, new_mask_2)
        else:
            for transformer in self.transformer_layers_1:
                x_1 = transformer(new_x_emb_1, new_mask_1) # [b, l, h]: LX
            for transformer in self.transformer_layers_1:
                x_2 = transformer(x, mask) # [b, l, h] : X
            for transformer in self.transformer_layers_1:
                x_3 = transformer(new_x_emb_2, new_mask_2) # [b, l, h] :GX
        mixed_x = x_1 + x_2 + x_3
        # print(x_2)
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
class AugmentationSelector(nn.Module):
    def __init__(self, max_seq_length):
        super(AugmentationSelector, self).__init__()
        self.max_seq_length = max_seq_length
    def forward(self, input_seq):
        # 计算每个序列的长度
        seq_lengths = torch.sum(input_seq != 0, dim=1).float()  # 忽略填充的部分
        # 归一化序列长度
        normalized_lengths = seq_lengths / self.max_seq_length
        # 将长度映射到insert和replace的概率
        probabilities = torch.sigmoid(normalized_lengths)
        return probabilities 

