import torch
from torch import nn
from layers.Transformer_EncDec import Endogenous_Encoder, Endogenous_encoder_layer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import pandas as pd
import os
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

def haversine(lat1, lon1, lats2, lons2):
    # Convert degrees to radians
    lat1, lon1 = np.array([lat1], dtype=np.float32), np.array([lon1], dtype=np.float32)
    lats2, lons2 = np.array(lats2, dtype=np.float32), np.array(lons2, dtype=np.float32)
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2, lons2 = np.radians(lats2), np.radians(lons2)

    # Haversine formula
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Earth radius in kilometers (you can change this to miles by using 3956)
    km = 6371
    return c * km

def calculate_normalized_weights(distances):
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-10
    # Calculate weights as the inverse of the distance
    inverse_weights = 1 / (np.array(distances, dtype=float) + epsilon)
    # Normalize the weights so they sum to 1 and are bound between 0 and 1
    normalized_weights = inverse_weights / np.sum(inverse_weights)
    return normalized_weights

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.endogenous_embedding = nn.Linear(self.seq_len, configs.d_model)
        # self.exdoegenous_embedding = nn.Linear(self.seq_len, configs.d_model)
        
        
        # Encoder
        self.encoder = Endogenous_Encoder(
            [
                Endogenous_encoder_layer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),    
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.feature_index = configs.feature_index
        self.exogenous_lst = configs.exogenous_lst

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2 + 1)

        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
            
            
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc shape: [64,168,28]
        # get the sub tensor of x_enc
        # x_enc = x_enc[:, :, :]
        x_enc = torch.cat((x_enc, x_mark_enc), dim=2)
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        x_endogenous = x_enc[:,[self.feature_index], :]
        x_exogenous = x_enc[:,self.exogenous_lst, :]
        enc_out, n_vars = self.patch_embedding(x_endogenous)
        x_endogenous = self.endogenous_embedding(x_endogenous)
        x_exogenous = self.endogenous_embedding(x_exogenous)
        
        enc_out = torch.cat((x_endogenous, enc_out), dim=1)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out, x_exogenous)
        
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

   
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]