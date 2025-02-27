import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.p = p = 3
        self.q = q = 2
        self.n_exog = n_exog = configs.enc_in - 1
        self.n_target = n_target = 1
        self.target_index = configs.feature_index
        self.pred_len = self.forecast_len = configs.pred_len
        self.task_name = configs.task_name

        # Constant term (intercept)
        self.constant = nn.Parameter(torch.zeros(n_target))
        # AR coefficients: one (n_target x n_target) matrix for each lag 1,...,p.
        self.phi = nn.Parameter(torch.Tensor(p, n_target, n_target))
        # MA coefficients: one (n_target x n_target) matrix for each lag 1,...,q.
        if q > 0:
            self.theta = nn.Parameter(torch.Tensor(q, n_target, n_target))
        else:
            self.theta = None
        # Exogenous coefficients: matrix mapping n_exog to n_target.
        # (We define beta as (n_exog, n_target) so that x_t (batch_size x n_exog)
        # multiplied by beta yields a (batch_size x n_target) vector.)
        self.beta = nn.Parameter(torch.Tensor(n_exog, n_target))
        
        # Initialize the parameters (using Xavier uniform initialization)
        nn.init.xavier_uniform_(self.phi)
        if self.q > 0:
            nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.beta)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_history= x_enc[:,:,[self.target_index]]
        # x_future = torch.cat((x_enc[:,:,:self.target_index], x_enc[:,:,self.target_index+1:]), dim=2)
        
        x_history= x_enc[:,:,[0]]
        x_future = x_enc[:,:,[1]]
        
        batch_size = x_history.shape[0]
        forecast_length = self.forecast_len
        # Start with the given history.
        history = x_history.clone()  # shape: (batch_size, p, n_target)
        forecasts = []
        
        # Forecast one step at a time.
        for t in range(forecast_length):
            # AR component: use the last p values from the history.
            ar_sum = 0.0
            for i in range(1, self.p + 1):
                y_lag = history[:, -i, :]  # (batch_size, n_target)
                ar_sum = ar_sum + torch.matmul(y_lag, self.phi[i-1].T)
            # Exogenous component: use the future input at time t.
            exog_sum = torch.matmul(x_future[:, t, :], self.beta)
            # In forecasting we set the MA part to zero.
            pred_t = self.constant + ar_sum + exog_sum
            forecasts.append(pred_t.unsqueeze(1))
            # Append the prediction to the history so that it can be used for subsequent forecasts.
            history = torch.cat([history, pred_t.unsqueeze(1)], dim=1)
        
        forecasts = torch.cat(forecasts, dim=1)
        forecasts.permute(0, 2, 1)
        return forecasts


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
