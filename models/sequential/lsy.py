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

        # self.hgnn_layer = HGNN(self.hidden_size)
        # self.LayerNorm = nn.LayerNorm(self.hidden_size) #, eps=self.layer_norm_eps)
        # self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # self.hg_type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        # self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # self.gating_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # # self.position_range = 3
        # self.sim_threshold = 0.1
        # self.max_sim_length =14
        
        # nn.init.normal_(self.attn, std=0.02)
        # nn.init.normal_(self.attn_weights, std=0.02)
        # # nn.init.normal_(self.gating_bias, std=0.02)
        # nn.init.normal_(self.gating_weight, std=0.02)
        # nn.init.normal_(self.metric_w1, std=0.02)
        # nn.init.normal_(self.metric_w2, std=0.02)

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

    def forward(self, batch_seqs):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers_1:
            x_1 = transformer(x, mask)
        for transformer in self.transformer_layers_2:
            x_2 = transformer(x, mask)
        for transformer in self.transformer_layers_3:
            x_3 = transformer(x, mask)
        return x_1 + x_2 + x_3

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        masked_seqs, masked_items = self._transform_train_seq(
            batch_seqs, batch_last_items.unsqueeze(1))
        # B, T, E
        logits = self.forward(masked_seqs) # [b, l]
        logits = self.out_fc(logits) # [b, l, n+1]
        # B, T, E -> B*T, E
        logits = logits.view(-1, logits.size(-1)) # [b*l, n+1]
        loss = self.loss_func(logits, masked_items.reshape(-1))
        loss_dict = {'rec_loss': loss.item()}
        return loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores = self.forward(masked_seqs)
        scores = self.out_fc(scores)
        scores = scores[:, -1, :]
        return scores