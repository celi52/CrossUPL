import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, self.pred_len)
        )
            

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        _, _, N = x_enc.shape
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        enc_out = self.encoder(enc_out)
        # get means over 2rd dimension and keep dims
        dec_out = torch.mean(enc_out, 1, keepdim=True).permute(0, 2, 1)
        
        # dec_out = enc_out.permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
