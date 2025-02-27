from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import wandb


warnings.filterwarnings('ignore')

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()  # Use PyTorch's built-in MSE loss

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

class CATSConfig:
    seq_len = 168                 # L_I, length of input
    pred_len = 24                 # L_P, forecasting horizon
    label_len = seq_len           # placeholder for predictors with encoder-decoder structure
    enc_in = 1                   # C, number of input series
    time_emb_dim = 4              # no timestamp embedding by default
    number_of_targets = 0         # partial OTS mode disabled: all the input series are considered as OTS
    features = 'M'
    
    F_conv_output = 10             # The output number of ATS channels, $n_m$, for $F^{[\mathtt{Conv}]}_m (X_I)$
    F_noconv_output = 10           # The output number of ATS channels, $n_m$, for $F^{[\mathtt{NOConv}]}_m (X_I)$
    F_gconv_output_rate = 1        # The output expanding rate of ATS channels, $v$, for $F^{[\mathtt{IConv}]}_{m} (X_I)$ with output channels $v \cdot C$
    F_lin_output = 10              # The output number of ATS channels, $n_m$, for $F^{[\mathtt{Lin}]}_m (X_I)$
    F_id_output_rate = 1           # The output expanding rate of ATS channels, $v$, for $F^{[\mathtt{Id}]}_{m} (X_I)$ with output channels $v \cdot C$
    F_emb_output = 10              # The output number of ATS channels, $n_m$, for $F^{[\mathtt{Emb}]}_m (X_I)$
    
    mlp_ratio = 1                  # shared MLP expanding ratio for different modules
    
    continuity_beta = 1            # weight of continuity loss
    temporal_gate = True           # whether to use temporal sparsity
    channel_sparsity = True        # whether to use channel sparsity
    
    predictor = 'Default'          # predictor name
    predictor_dropout = 0.1

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        self.pjm_sub_name = args.pjm_sub_name
        self.model_id = args.model_id
        
    def _build_model(self):
        if self.args.model != 'CATS':
            model = self.model_dict[self.args.model].Model(self.args).float()
        else:
            model = self.model_dict[self.args.model].Model(CATSConfig).float()
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'L1':
            criterion = nn.L1Loss()
        elif self.args.loss == 'RMSE':
            criterion = RMSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_size, day_dim, hour_dim, feature_dim = batch_x.shape
                _, label_len, _, _ = batch_y.shape
                
                batch_x = batch_x.reshape(batch_size, day_dim * hour_dim, feature_dim)
                batch_y = batch_y.reshape(batch_size, label_len * hour_dim, feature_dim)
                
                mark_dim = batch_x_mark.shape[-1]
                batch_x_mark = batch_x_mark.reshape(batch_size, day_dim * hour_dim, mark_dim)
                batch_y_mark = batch_y_mark.reshape(batch_size, label_len * hour_dim, mark_dim)
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = self.args.feature_index if self.args.features == 'MS' and self.args.model != 'ARIMAX' else 0
                outputs = outputs[:, -self.args.pred_len:, [f_dim]]
                batch_y = batch_y[:, -self.args.pred_len:, [f_dim]].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)
                if torch.isnan(loss).any():
                    print('ok')

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_size, day_dim, hour_dim, feature_dim = batch_x.shape
                _, label_len, _, _ = batch_y.shape
                
                batch_x = batch_x.reshape(batch_size, day_dim * hour_dim, feature_dim)
                batch_y = batch_y.reshape(batch_size, label_len * hour_dim, feature_dim)
                
                mark_dim = batch_x_mark.shape[-1]
                batch_x_mark = batch_x_mark.reshape(batch_size, day_dim * hour_dim, mark_dim)
                batch_y_mark = batch_y_mark.reshape(batch_size, label_len * hour_dim, mark_dim)
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = 0 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, [f_dim]]
                        batch_y = batch_y[:, -self.args.pred_len:, [f_dim]].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = self.args.feature_index if self.args.features == 'MS' and self.args.model != 'ARIMAX' else 0
                    outputs = outputs[:, -self.args.pred_len:, [f_dim]]
                    batch_y = batch_y[:, -self.args.pred_len:, [f_dim]].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_size, day_dim, hour_dim, feature_dim = batch_x.shape
                _, label_len, _, _ = batch_y.shape
                

                batch_x = batch_x.reshape(batch_size, day_dim * hour_dim, feature_dim)
                batch_y = batch_y.reshape(batch_size, label_len * hour_dim, feature_dim)
                
                mark_dim = batch_x_mark.shape[-1]
                batch_x_mark = batch_x_mark.reshape(batch_size, day_dim * hour_dim, mark_dim)
                batch_y_mark = batch_y_mark.reshape(batch_size, label_len * hour_dim, mark_dim)
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # outputs = torch.mean(outputs, dim=1)
                f_dim = self.args.feature_index if self.args.features == 'MS' and self.args.model != 'ARIMAX' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, [f_dim]]
                batch_y = batch_y[:, :, [f_dim]]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))
        f = open("result_" + self.model_id + ".txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        return
