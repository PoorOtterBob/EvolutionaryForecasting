from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, MSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
import os
import time
from datetime import datetime
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import pandas as pd

from exp.exp_grad_check import MultiScaleGradientAnalyzer, plot_analysis_results
import sys

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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
        criterion = nn.MSELoss()
        return criterion
 
    def _model(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        # Forecasting Paradigm
        cot_step = math.ceil((batch_y.shape[1]-self.args.label_len) / self.model.pred_len)
        
        if cot_step == 1: # DF
             # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, :self.args.pred_len]
        else:  # CoT
            preds_list = []
            # output = torch.zeros_like(dec_inp[:, self.args.label_len:]).to(self.device)
            marks = torch.cat([batch_x_mark, batch_y_mark[:, self.args.label_len:]], dim=1)
            inputs = F.pad(batch_x, (0, 0, 0, cot_step*self.model.pred_len), "constant", 0)
            for step in range(cot_step):
                input_index_start = step*self.model.pred_len
                output_index_start = step*self.model.pred_len + self.model.seq_len

                input_data = inputs[:, input_index_start:output_index_start]
                input_mark = marks[:, input_index_start:output_index_start]
                output_mark = marks[:, output_index_start-self.args.label_len:output_index_start+self.model.pred_len]

                dec_inp = torch.zeros_like(batch_y[:, -self.model.pred_len:, :]).float()
                dec_inp = torch.cat([
                    batch_y[:, input_index_start:input_index_start+self.args.label_len], 
                    dec_inp
                    ], dim=1).float().to(self.device)
                pred = self.model(input_data, input_mark, dec_inp, output_mark)
                preds_list.append(pred)
                inputs[:, output_index_start:output_index_start+self.model.pred_len] = inputs[:, output_index_start:output_index_start+self.model.pred_len] + preds_list[-1]
            outputs = torch.cat(preds_list, dim=1)[:, :batch_y.shape[1]-self.args.label_len]
        return outputs

    def _model_train(self, batch_x, batch_x_mark, batch_y, batch_y_mark, k_start=None, sampling_prob=0.0): 
        return self._model(self, batch_x, batch_x_mark, batch_y, batch_y_mark)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    outputs = self._model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        origin_pred_len = self.args.pred_len
        if (self.args.check_pred_len != 0) and (self.args.eftrain):
            self.args.pred_len = self.args.check_pred_len
            print('Train horizon {} and model horizon {}.'.format(self.args.pred_len, self.model.pred_len))
        train_data, train_loader = self._get_data(flag='train')
        self.args.pred_len = origin_pred_len
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
        
        
        
        all_epochs_history = []
        for epoch_idx, epoch in enumerate(range(self.args.train_epochs)):
            iter_count = 0
            train_loss = []

            batch_ratios = []
            batch_matrices = []

            
            prob = min(1.0, (epoch_idx % 5) * 0.2) 

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                
                total_blocks = math.ceil((batch_y.shape[1]-self.args.label_len) / self.model.pred_len)
                k_start = total_blocks
                # 动态调整 k_start (逆向阶梯)
                # 每 5 个 epoch，k_start 向前移动一步
                if epoch_idx > 0 and epoch_idx % 5 == 0:
                    k_start = max(1, k_start - 1)   

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.eftrain:
                            outputs = self._model_train(batch_x, batch_x_mark, batch_y, batch_y_mark, k_start, prob)
                        else:
                            outputs = self._model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.eftrain:
                        outputs = self._model_train(batch_x, batch_x_mark, batch_y, batch_y_mark, k_start, prob)
                    else:
                        outputs = self._model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.checkgrad: 
                    analyzer = MultiScaleGradientAnalyzer(self.model)
                    # df_segs = analyzer.analyze_segments(batch_x, batch_y)
                    # Analyze single batch
                    ratios, matrix =  analyzer.analyze_segments([batch_x, batch_x_mark, None, None], batch_y)
                    
                    batch_ratios.append(ratios)
                    batch_matrices.append(matrix)
                    # df_snr = analyzer.analyze_snr_trend(batch_x, batch_y, step_stride=1) # 步长设大点跑得快
                    # plot_analysis_results(df_segs, df_snr, self.args.model, self.args.dataname)
                    # print('Done!!')
                    # sys.exit(0)
            

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
            ######
            if self.args.checkgrad:
                # --- Aggregation ---
                print(f"\n>>> Aggregating Statistics for Epoch {epoch_idx}...")
                
                # 1. Aggregate Norm Ratios
                # Convert list of dicts to DataFrame
                df_ratios = pd.DataFrame(batch_ratios)
                ratio_mean = df_ratios.mean().to_dict()
                ratio_std = df_ratios.std().to_dict()
                
                # 2. Aggregate Pairwise Matrices
                # Stack into [N_batches, 5, 5]
                all_matrices = np.array(batch_matrices)
                matrix_mean = np.mean(all_matrices, axis=0)
                matrix_std = np.std(all_matrices, axis=0)
                
                # Keys for matrix columns/rows
                seg_keys = ["96", "192", "336", "720", "All"]
                
                # --- Formatting Outputs ---
                
                # Print Norm Ratios Table
                print(f"\n[Epoch {epoch_idx}] Signal Masking (Norm Ratio vs Total)")
                print(f"{'Segment':<10} | {'Mean Ratio':<15} | {'Std':<15}")
                print("-" * 45)
                for k in ["96", "192", "336", "720"]:
                    print(f"{k:<10} | {ratio_mean[k]:<15.4f} | {ratio_std[k]:<15.4f}")
                    
                # Print Pairwise Matrix (Mean)
                df_matrix_mean = pd.DataFrame(matrix_mean, index=seg_keys, columns=seg_keys)
                print(f"\n[Epoch {epoch_idx}] Pairwise Similarity (Mean)")
                print(df_matrix_mean.round(4))
                
                # Print Pairwise Matrix (Std)
                df_matrix_std = pd.DataFrame(matrix_std, index=seg_keys, columns=seg_keys)
                print(f"\n[Epoch {epoch_idx}] Pairwise Similarity (Std)")
                print(df_matrix_std.round(4))

                # Prepare Return Data Structure
                epoch_stats = {
                    "epoch": epoch_idx,
                    "ratios": {"mean": ratio_mean, "std": ratio_std},
                    "matrix": {"mean": df_matrix_mean, "std": df_matrix_std}
                }
                all_epochs_history.append(epoch_stats)
                

            #######
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
        
        if self.args.checkgrad:
            torch.save(all_epochs_history, 
                    '/root/autodl-tmp/TSL/draw/'+self.args.model+'_'+self.args.dataname+'.pth')
            print('DONE!!!!!!!!!!')
            sys.exit(0)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        if self.args.check_pred_len != 0:
            self.args.pred_len = self.args.check_pred_len
        test_data, test_loader = self._get_data(flag='test')
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        outputs = self._model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    outputs = self._model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        text_format = 'results_txt/{}_{}_Seq{}_Pred{}_result_long_term_forecast.txt'.format(
            self.args.model,
            self.args.dataname,
            self.model.seq_len,
            self.model.pred_len,
        )
        f = open(text_format, 'a')
        # f.write(setting + "  \n")

        if trues.shape[-2] in [96, 192, 336, 720]:
            for length in [96, 192, 336, 720]:
                if trues.shape[-2] == length:
                    mae, mse, rmse, mape, mspe = metric(preds[:, :length], trues[:, :length])
                    text = '{} Horizon {:03d}: MSE, MAE {:.3f} {:.3f}, dtw:{} (Model Seq {:03d}, Pred {:03d})'.format(datetime.now(), 
                                                                                    length, 
                                                                                    mse, 
                                                                                    mae, 
                                                                                    dtw,
                                                                                    self.model.seq_len,
                                                                                    self.model.pred_len
                                                                                    )
                    print(text)  
                    f.write(text)
                    f.write('\n')
                    f.write('\n')
        else:
            length = trues.shape[-2]
            mae, mse, rmse, mape, mspe = metric(preds[:, :length], trues[:, :length])
            text = '{} Horizon {:03d}: MSE, MAE {:.3f} {:.3f}, dtw:{} (Model Seq {:03d}, Pred {:03d})'.format(datetime.now(), 
                                                                            length, 
                                                                            mse, 
                                                                            mae, 
                                                                            dtw,
                                                                            self.model.seq_len,
                                                                            self.model.pred_len
                                                                            )
            print(text)  
            f.write(text)
            f.write('\n')
            f.write('\n')
        f.close()

        if self.args.check_grad:
            for i in range(trues.shape[1]):
                print(MSE(preds[:, i], trues[:, i]))

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
