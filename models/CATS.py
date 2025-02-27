import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.eps = 1e-7
        ########## Input Data Format ########## 
        # seq_len: L_I, pred_len: L_P
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # in_channels: C
        self.in_channels = configs.enc_in
        # time_emb_dim: number of dimension of the optional timestamp embeddings
        self.time_emb_dim = configs.time_emb_dim
        # total number of input channels
        self.in_channels_full = self.in_channels + self.time_emb_dim
        
        # If you are only using a subset of the input series as the OTS and treating the remaining input series as ATS, you can set `number_of_targets` to a value greater than 0.
        if configs.number_of_targets > 0:
            self.preserving_channels = configs.number_of_targets
        else:
            self.preserving_channels = self.in_channels

        ########## Preprocessing ##########
        # Change the preprocessing method for input and output here, last-value demeaning by default
        self.preprocessing_method = 'lastvalue' #'standardization'#'auel'#'none'
        self.preprocessing_detach = True
        self.preprocessing_affine = False
        self.channel_w = nn.Parameter(torch.ones(self.preserving_channels), requires_grad=True)
        self.channel_b = nn.Parameter(torch.zeros(self.preserving_channels), requires_grad=True)
        if self.preprocessing_method == 'auel':
            self.conv_size = [8, 16, 64]
            self.ema_alpha = nn.Parameter(0.9*torch.ones(self.in_channels))
            self.std_weights = nn.Parameter(torch.cat([0.5*torch.ones(len(self.conv_size), self.in_channels), 
                                                          torch.ones(1, self.in_channels)]))

        ########## ATS Constructors (OTS -> ATS) ##########
        # The output number of ATS channels, $n_m$, for $F^{[\mathtt{Conv}]}_m (X_I)$
        self.F_conv_output = configs.F_conv_output
        # The output number of ATS channels, $n_m$, for $F^{[\mathtt{NOConv}]}_m (X_I)$
        self.F_noconv_output = configs.F_noconv_output
        # The output expanding rate of ATS channels, $v$, for $F^{[\mathtt{IConv}]}_{m} (X_I)$ with output channels $v \cdot C$
        self.F_gconv_output_rate = configs.F_gconv_output_rate
        # The output number of ATS channels, $n_m$, for $F^{[\mathtt{Lin}]}_m (X_I)$
        self.F_lin_output = configs.F_lin_output
        # The output expanding rate of ATS channels, $v$, for $F^{[\mathtt{Id}]}_{m} (X_I)$ with output channels $v \cdot C$
        self.F_id_output_rate = configs.F_id_output_rate
        # The output number of ATS channels, $n_m$, for $F^{[\mathtt{Emb}]}_m (X_I)$
        self.F_emb_output = configs.F_emb_output
        # calculate the $n_m$ for IConv and Id
        self.F_gconv_output = int(self.F_gconv_output_rate*self.preserving_channels)
        self.F_id_output = int(self.preserving_channels*self.F_id_output_rate)

        # The components of ATS channels. Note that we are using two types of Conv and NOConv with different kernel size 
        self.feature_pool_desc = [self.F_conv_output, 
                                  self.F_conv_output, 
                                  self.F_noconv_output, 
                                  self.F_noconv_output, 
                                  self.F_gconv_output, 
                                  self.F_lin_output, 
                                  self.F_id_output, 
                                  self.F_emb_output,
                                  self.time_emb_dim,
                                  self.in_channels
                                 ]

        # shared MLP upscaling ratio in channel sparsity module and predictors
        self.mlp_ratio = configs.mlp_ratio


        # The two Conv ATS constructors $F^{[\mathtt{Conv}]}_1 (X_I)$ and $F^{[\mathtt{Conv}]}_2 (X_I)$
        self.input_conv2 = nn.Conv1d(in_channels=self.in_channels_full, out_channels=self.F_conv_output, kernel_size=49, padding=24)
        self.input_conv2_alter = nn.Conv1d(in_channels=self.in_channels_full, out_channels=self.F_conv_output, kernel_size=193,padding=96)
        # The two NOConv ATS constructors $F^{[\mathtt{NOConv}]}_3 (X_I)$ and $F^{[\mathtt{NOConv}]}_4 (X_I)$
        self.input_conv = Conv1dNonOverlapping(in_channels=self.in_channels_full, out_channels=self.F_noconv_output, input_length=self.seq_len, kernel_size=12, groups=1)
        self.input_conv_alter = Conv1dNonOverlapping(in_channels=self.in_channels_full, out_channels=self.F_noconv_output, input_length=self.seq_len, kernel_size=24, groups=1)
        # Independent Convolution
        self.input_conv_grouped2 = nn.Conv1d(in_channels=self.preserving_channels, out_channels=self.F_gconv_output, kernel_size=49, padding=24, groups=self.preserving_channels)
        # Linear Projection
        self.input_conv_1x1 = nn.Conv1d(in_channels=self.in_channels_full, out_channels=self.F_lin_output, kernel_size=1, padding=0)
        # Embedding
        self.pure_emb = nn.Parameter(torch.randn(1, self.seq_len, self.F_emb_output))
        self.dropout = nn.Dropout1d(0.1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        ########## Main Predictor ##########
        # which predictor to use
        self.predictor = configs.predictor
        print(f'Current Predictor: {self.predictor}')

        # A interface to use different main predictor in CATS is provided here. By changing the predictor_configs, you can personalize the hyper-parameters for different predictors. 
        # The implementation of the main predictors will be provided in the following sections.
        # In the first part of this notebook, we will use the implementation of a 2-layer MLP, the `DefaultPredictor` here, as an example.
        # To customize your own predictor, you can write your code as the `DefaultPredictor` class in the next block and add one if-statement here
        predictor_configs = copy.deepcopy(configs)
        predictor_configs.enc_in = sum(self.feature_pool_desc)
        predictor_configs.dec_in = sum(self.feature_pool_desc)
        predictor_configs.c_out = sum(self.feature_pool_desc)
        if self.predictor == 'Default':
            self.Predictor = DefaultPredictor(d_in=self.seq_len, d_ff=self.mlp_ratio*self.seq_len, d_out=self.pred_len, dropout=configs.predictor_dropout)
        
        # We will show examples of other predictors in the later part of this notebook
        elif self.predictor == 'Simple':
            self.Predictor = SimplePredictor(d_in=self.seq_len, d_out=self.pred_len)
        elif self.predictor == 'MLP':
            self.Predictor = MLPPredictor(d_in=self.seq_len, d_ff=self.mlp_ratio*(self.seq_len+self.pred_len), d_out=self.pred_len, dropout=configs.predictor_dropout)
        elif self.predictor == 'NLinear':
            self.Predictor = LinearPredictor(d_in=self.seq_len, d_out=self.pred_len)
        elif self.predictor == 'Agg':
            self.Predictor = AggregatingPredictor(d_in=self.seq_len, d_out=self.pred_len, n_channels=predictor_configs.enc_in, kernel_size=32, token_size=64, dropout=0.5, independent=True)
        elif self.predictor == 'Mean':
            self.Predictor = MeanAggregationPredictor(pred_len=self.pred_len)
        elif self.predictor == 'IndependentLinear':
            self.Predictor = IndependentLinearPredictor(d_in=self.seq_len, d_out=self.pred_len, n_channels=predictor_configs.enc_in, split_ratio=1)
        elif self.predictor == 'GroupedIndependent':
            self.Predictor = GroupedLinearPredictor(d_in=self.seq_len, d_out=self.pred_len, grouping=self.feature_pool_desc[0:-1] + [1 for i in range(self.in_channels)])
        elif self.predictor == 'MoE':
            self.Predictor = MoEPredictor(d_in=self.seq_len, d_ff=4096, d_out=self.pred_len, dropout=configs.predictor_dropout, n_experts=4)

        # The predictors below are not included in this notebook, please refer to their corresponding original code repository for the implementation
        elif self.predictor == 'PatchTST':
            predictor_configs.e_layers = 3
            predictor_configs.n_heads = 4
            predictor_configs.d_model = 16
            predictor_configs.d_ff = 128
            predictor_configs.dropout = 0.3
            predictor_configs.fc_dropout = 0.3
            predictor_configs.head_dropout = 0
            predictor_configs.patch_len = 16
            predictor_configs.stride = 8
            predictor_configs.revin = 1
            self.Predictor = PatchTST(predictor_configs)
        elif self.predictor == 'DLinear':
            self.Predictor = DLinear(predictor_configs)
        elif self.predictor == 'Autoformer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Autoformer(predictor_configs)
        elif self.predictor == 'Informer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Informer(predictor_configs)
        elif self.predictor == 'TimesNet':
            predictor_configs.label_len = 48
            predictor_configs.factor = 3
            self.Predictor = TimesNet(predictor_configs)
        elif self.predictor == 'Transformer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Transformer(predictor_configs)
        elif self.predictor == 'TiDE':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 2
            predictor_configs.label_len = 48
            predictor_configs.factor = 3
            predictor_configs.d_model = 256
            predictor_configs.d_ff = 256
            predictor_configs.dropout = 0.3
            self.Predictor = TiDE(predictor_configs)
        elif self.predictor == 'NBeats':
            self.Predictor = NBeats(predictor_configs)
        elif self.predictor == 'FITS':
            predictor_configs.individual = 0
            self.Predictor = FITS(predictor_configs)
        self.predictor_configs = predictor_configs

        ########## Three Key Principles ##########
        # $\mathcal{L}_{\text{cont}}$
        self.continuity_beta = configs.continuity_beta

        # Whether to use temporal sparsity module: True or False
        self.gated = configs.temporal_gate
        # independent linear layers in temporal sparsity module
        self.generate_self_mask = GroupedLinears(self.seq_len, 1, n=predictor_configs.enc_in, zero_init=True, init=0)
        self.mask_generate_linspace = nn.Parameter(torch.linspace(1, 0, steps=self.seq_len), requires_grad=False)

        # Whether to use channel sparsity module: True or False
        self.channel_sparsity = configs.channel_sparsity
        # channel sparsity module
        self.channel_selection = SelectiveChannelModule(input_length=self.seq_len, input_channels=self.in_channels_full, 
                                                        expanded_channels=sum(self.feature_pool_desc)+self.time_emb_dim, 
                                                        drop_channels=0, mlp_ratio=self.mlp_ratio)

        ########## ATS -> OTS ##########
        # Project ATS back to OTS channels
        self.output_conv_1x1 = nn.Conv1d(in_channels=predictor_configs.dec_in+self.time_emb_dim, out_channels=self.preserving_channels, kernel_size=1, padding=0)
        torch.nn.init.zeros_(self.output_conv_1x1.weight)

    # Implementation of temporal sparsity module
    def gate(self, x, generated_mask, linspace, return_mask=False):
        mask_infer = -generated_mask(x) # B 1 C, a, a \in (-inf, 0)
        gate_len = x.shape[1]
        mask_infer = linspace.reshape(1, gate_len, 1).repeat(mask_infer.shape[0], 1, mask_infer.shape[-1]) * mask_infer + 1 # a(t-L_I)+1
        # straigh through estimator
        mask_stop = mask_infer - mask_infer.detach() + (mask_infer > 0).int().detach()
        
        # save mask_stop for visualization
        # torch.save(mask_stop, 'AE_combine.pt')
        
        x = x * mask_stop
        if return_mask:
            return x, mask_stop
        else:
            return x

    # ATS construction
    def encoder(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x = torch.cat([
            self.input_conv2(x.permute(0,2,1)).permute(0,2,1) if self.F_conv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv2_alter(x.permute(0,2,1)).permute(0,2,1) if self.F_conv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv(x.permute(0,2,1)).permute(0,2,1) if self.F_noconv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv_alter(x.permute(0,2,1)).permute(0,2,1) if self.F_noconv_output > 0 else torch.Tensor([]).to(x.device),
            self.input_conv_grouped2(x[:, :, -self.preserving_channels:].permute(0,2,1)).permute(0,2,1) if self.F_gconv_output_rate > 0 else torch.Tensor([]).to(x.device),
            self.input_conv_1x1(x.permute(0,2,1)).permute(0,2,1) if self.F_lin_output > 0 else torch.Tensor([]).to(x.device),
            *[x[:, :, -self.preserving_channels:] for i in range(self.F_id_output_rate)],
            self.pure_emb.repeat(x.shape[0], 1, 1),
            x
        ], dim=-1) # B, L_I, total_ATS_channels+C
        
        return x

    # Main predictor
    def predictor_(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_pred = self.Predictor(x.clone(), x_mark_enc, x_dec, x_mark_dec)
        x = torch.cat([x.clone(), x_pred], dim=1) # B L latent
        return x

    # Project ATS channels back to OTS channels, maintaining OTS shortcut
    def decoder(self, x, channel_attn_score_decode):
        x_resid = self.output_conv_1x1((x*channel_attn_score_decode).permute(0,2,1)).permute(0,2,1) # B L C
        output = x.clone()
        output[:, -self.pred_len:, -self.preserving_channels:] = x[:, -self.pred_len:, -self.preserving_channels:] + x_resid[:, -self.pred_len:, :]
        return output

    # Implementation of Last-value Demeaning / RevIn / AUEL. Last-value Demeaning is used by default.
    def preprocessing(self, x, x_output=None, cutoff=None, preprocessing_method=None, preprocessing_detach=None, preprocessing_affine=None):
        eps = self.eps
        x_orig = x.clone()
        if preprocessing_method is None: preprocessing_method = self.preprocessing_method
        if preprocessing_detach is None: preprocessing_detach = self.preprocessing_detach
        if preprocessing_affine is None: preprocessing_affine = self.preprocessing_affine
        if self.preprocessing_affine:
            x = x * self.channel_w + self.channel_b
        if preprocessing_method == 'lastvalue':
            mean = x[:,-1:,:]
            std = torch.ones(1, 1, x.shape[-1]).to(x.device)
        elif preprocessing_method == 'standardization':
            if cutoff is None:
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True) + eps
            else:
                n_elements = cutoff.sum(dim=1, keepdim=True) # B L C -> B 1 C
                # x = x * cutoff # already done
                mean = x.sum(dim=1, keepdim=True) / (n_elements+eps).detach()
                sst = (cutoff*((torch.square(x - mean))**2)).sum(dim=1, keepdim=True)
                std = torch.sqrt(sst / (n_elements-1+eps)) + eps
                std = std.detach()
                std = torch.ones(1, 1, x.shape[-1]).to(x.device)
        elif preprocessing_method == 'auel':
            anchor_context = ema_3d(x[:, 0:(self.seq_len), :], alpha=self.ema_alpha[0:x.shape[-1]].clip(0.000001, 0.999999))
            std = weighted_std(x[:, 0:(self.seq_len), :], self.conv_size+[self.seq_len], gate_activation(self.std_weights*2).unsqueeze(0).repeat(x.shape[0], 1, 1)[:, :, 0:x.shape[-1]]).clip(1e-4, 5000)+eps
            mean = anchor_context
        elif preprocessing_method == 'none':
            mean = torch.zeros(1, 1, x.shape[-1]).to(x.device)
            std = torch.ones(1, 1, x.shape[-1]).to(x.device)
        if x_output is None:
            x_output = x_orig.clone()
        if self.preprocessing_detach:
            x_output = (x_output - mean.detach())/std.detach()
        else:
            x_output = (x_output - mean)/std
        if self.preprocessing_affine:
            x_output = (x_output - self.channel_b) / self.channel_w
        return x_output, mean, std

    # Inverse processing stage of the preprocessing method used above
    def inverse_processing(self, x, mean, std):
        x = x.clone()
        if self.preprocessing_affine:
            x = (x - self.channel_b)/self.channel_w
        x = x*std + mean
        return x

    # In this example, only the input OTS `x_enc` is needed
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, 
                return_mask=False, train=False):
        # x_enc: [Batch, Input length, Channel]
        # Apply preprocessing method
        x_preprocessed, mean, std = self.preprocessing(x_enc[:, :, -self.preserving_channels:], x_output=x_enc[:, :, -self.preserving_channels:])
        x = torch.cat([x_enc[:, :, 0:-self.preserving_channels], x_preprocessed], dim=-1)
        
        # Add additional timestamp embedding if applicable (not used in this notebook example)
        if self.time_emb_dim:
            x_enc = torch.cat([x_mark_enc, x], dim=-1)
        else:
            x_enc = x

        # OTS -> ATS

        # Calculate the channel attention score with the channel sparsity module, for both ATS and OTS
        channel_attn_score = self.channel_selection(x_enc) if self.channel_sparsity else torch.ones(1, 1, sum(self.feature_pool_desc)+self.time_emb_dim, device=x_enc.device)
        channel_attn_score_decode = torch.cat([torch.ones_like(channel_attn_score[:, :, 0:-(self.preserving_channels+self.time_emb_dim)]).to(x.device), channel_attn_score[:, :, -(self.preserving_channels+self.time_emb_dim):]], dim=-1).clone()
        channel_attn_score = torch.cat([channel_attn_score[:, :, 0:-(self.preserving_channels+self.time_emb_dim)], torch.ones(channel_attn_score.shape[0], 1, self.preserving_channels, device=x_enc.device)], dim=-1)
        
        # ATS construction
        x = self.encoder(x_enc)

        # Apply channel attention score
        x = channel_attn_score * x

        # Calculate continuity loss
        cnt_loss = continuity_loss(x)*self.continuity_beta

        # Apply temporal sparsity
        if return_mask:
            x, mid_mask = self.gate(x, self.generate_self_mask, self.mask_generate_linspace, return_mask=True)
        else:
            x = self.gate(x, self.generate_self_mask, self.mask_generate_linspace)

        # Some predictors require zero-filled decoder input:
        # Align Input Format for Different Predictors
        x_dec = torch.zeros(x.shape[0], self.pred_len, x.shape[-1], device=x.device)
        if self.predictor_configs.label_len:
            x_dec = torch.cat([x[:, -self.predictor_configs.label_len:, :], x_dec], dim=1)

        # Predictor
        x = self.predictor_(x, x_mark_enc, x_dec, x_mark_dec)

        # Take snapshot for the visualization of ATS 
        x_predictor_snapshot = x.detach().clone()
        x_predictor_snapshot[:, :, -self.preserving_channels:] = self.inverse_processing(x_predictor_snapshot[:, :, -self.preserving_channels:], mean, std).detach()

        # Add Timestep embedding again, if applicable (not used in this notebook example)
        if self.time_emb_dim:
            timestamp_embedding = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
            x = torch.cat([timestamp_embedding, x], dim=-1)

        # Decoder
        x_dec = self.decoder(x, channel_attn_score_decode)

        # Inverse processing stage of the preprocessing method
        x = x_dec[:, -self.pred_len:, -self.in_channels:]
        x[:, :, -self.preserving_channels:] = self.inverse_processing(x[:, :, -self.preserving_channels:], mean, std)

        if train:
            return x, cnt_loss

        # for visualization
        if return_mask:
            return x, mid_mask, x_predictor_snapshot
        else:
            return x # [Batch, Output length, Channel]
        

# Building Key Modules]

# Non-overlapping Convolution
class Conv1dNonOverlapping(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, groups=1):
        super(Conv1dNonOverlapping, self).__init__()
        extra_length = (kernel_size - (input_length % kernel_size)) % kernel_size
        padding = extra_length // 2
        self.input_length = input_length
        self.groups = groups
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels*kernel_size, kernel_size=kernel_size, stride=kernel_size, padding=padding, groups=groups)
        
    def forward(self, x):
        # x: B, C, L
        x = self.conv1d(x)
        conv_len = x.shape[-1]
        x = x.reshape(x.shape[0], self.out_channels, self.kernel_size, conv_len)  # B G*C Ks Lout
        x = x.permute(0, 1, 3, 2) # B G*C Lout Ks
        x = x.reshape(x.shape[0], self.out_channels, conv_len*self.kernel_size) # B G*C Lout*Ks
        padding = self.input_length - x.shape[-1]
        x = F.pad(x, (padding, 0))
        return x

# Implementation of Parallelized Independent Linear Layers
class GroupedLinears(nn.Module):
    def __init__(self, in_d, out_d, n=16, zero_init=True, init=0, split_ratio=1):
        super(GroupedLinears, self).__init__()
        self.n = n
        self.split_ratio = min(split_ratio, n)
        base_segment_size = n // self.split_ratio
        remainder = n % self.split_ratio
        self.segment_sizes = [base_segment_size + (1 if i < remainder else 0) for i in range(self.split_ratio)]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.weights1 = nn.ParameterList()
        self.biases1 = nn.ParameterList()
        
        for segment_size in self.segment_sizes:
            if zero_init:
                self.weights.append(nn.Parameter(torch.zeros(segment_size, in_d, out_d)))
                self.biases.append(nn.Parameter(torch.zeros(segment_size, 1, out_d)))
            else:
                self.weights.append(nn.Parameter(nn.init.trunc_normal_(torch.zeros(segment_size, in_d, out_d), mean=0.0, std=0.01, a=-0.02, b=0.02)))
                self.biases.append(nn.Parameter(torch.zeros(segment_size, in_d, out_d)))
                self.weights1.append(nn.Parameter(torch.zeros(segment_size, in_d, out_d)))
                self.biases1.append(nn.Parameter(torch.zeros(segment_size, 1, out_d)))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.init = init

    def forward(self, x):
        # Input: B L C
        outputs = []
        start = 0
        for i in range(self.split_ratio):
            end = start + self.segment_sizes[i]
            x_segment = x[:, :, start:end]  # B L C_segment
            x_segment = x_segment.permute(0, 2, 1).unsqueeze(2)  # B C_segment 1 L
            w = self.weights[i].unsqueeze(0)  # 1 C_segment L Lp
            b = self.biases[i].unsqueeze(0)  # 1 C_segment L Lp
            x_segment = torch.matmul(x_segment, w) + b  # B C_segment 1 Lp
            x_segment = x_segment[:, :, 0, :]  # B C_segment Lp
            x_segment = x_segment.permute(0, 2, 1)  # B Lp C_segment
            outputs.append(x_segment)
            start = end
        x = torch.cat(outputs, dim=2)  # B Lp C
        x = x + self.init
        return x

# Implementation of the Channel Sparsity Module
class SelectiveChannelModule(nn.Module):
    def __init__(self, input_length, input_channels, expanded_channels, drop_channels=1, mlp_ratio=2):
        super(SelectiveChannelModule, self).__init__()
        self.input_channels = input_channels # C
        self.expanded_channels = expanded_channels # N
        self.drop_channels = drop_channels # C
        self.attention = nn.Sequential(
            nn.Linear(input_channels, int(mlp_ratio*(expanded_channels))),
            nn.GELU(),
            nn.Linear(int(mlp_ratio*(expanded_channels)), expanded_channels),
            nn.Sigmoid()
        )
        # original channel sparsity module
        self.aggregation = nn.Sequential(
            nn.Linear(input_length, 1),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.padding = nn.Parameter(torch.ones(1, 1, drop_channels), requires_grad=False)
    
    def forward(self, x):
        # x: Batch L C
        attention_scores = self.aggregation(x.permute(0,2,1)).permute(0,2,1) # Batch 1 C
        attention_scores = self.attention(attention_scores) # Batch 1 N
        # OTS are not affected in the channel sparsity. Their attention scores are set to 1.
        if self.drop_channels > 0:
            attention_scores = torch.cat([attention_scores[:, :, 0:-self.drop_channels], self.padding.repeat(x.shape[0], 1, 1)], dim=-1)
        return attention_scores


# Continuity Loss
def continuity_loss(tensor):
    B, L, C = tensor.shape
    mean = torch.mean(tensor, dim=1, keepdim=True)
    std = torch.std(tensor, dim=1, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + 1e-6)
    diffs = normalized_tensor[:, 1:, :] - normalized_tensor[:, :-1, :]
    squared_diffs = diffs ** 2
    loss = torch.mean(squared_diffs)
    return loss


# Building Default Predictor
# Implementation of Default Predictor: 2-Layer MLP
class DefaultPredictor(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0.1):	
        super(DefaultPredictor, self).__init__()
        self.model = nn.Sequential(nn.Linear(d_in, d_ff, bias=True),
                                   nn.Dropout(dropout),
                                   nn.GELU(),
                                   nn.Linear(d_ff, d_out, bias=True))
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):	
        # Input x: (B L C)
        x = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    
    
    