import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.use_day_index = configs.use_day_index if configs.hour_length > 0 else 0
        self.use_hour_index = configs.use_hour_index if configs.day_length > 0 else 0
        self.scale = 0.02
        self.emb_len_hour = configs.hour_length
        self.emb_len_day = configs.day_length
        layers = [
                nn.Linear(self.seq_len, self.d_model),
                # nn.Linear(self.seq_len // 2 + 1, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
        ]
        self.model = nn.Sequential(*layers)
        
        self.emb_hour = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
    
        self.emb_day = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

    def _index(self, x_mark_enc):
        hour_index = None
        day_index = None

        if self.use_hour_index:
            if self.emb_len_hour == 24:
                hour_index = torch.round((x_mark_enc[:, -1, 0:1]+0.5)*(24-1))
            elif self.emb_len_hour == 48:
                hour_index = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*4+\
                                    torch.round((x_mark_enc[:, -1, 1:2]+0.5)*(60-1))/30
            elif self.emb_len_hour == 96:
                hour_index = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*4+\
                                    torch.round((x_mark_enc[:, -1, 1:2]+0.5)*(60-1))/15
            elif self.emb_len_hour == 144:
                hour_index = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*6+\
                                    torch.round((x_mark_enc[:, -1,  1:2]+0.5)*(60-1))/10
            elif self.emb_len_hour == 288:
                hour_index = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*12+\
                                    torch.round((x_mark_enc[:, -1, 1:2]+0.5)*(60-1))/5         
        
        if self.use_day_index:
            if x_mark_enc.shape[-1] == 4: 
                day_index = (x_mark_enc[..., -1, 1]+0.5)*(7-1)
            elif x_mark_enc.shape[-1] == 6:
                day_index = (x_mark_enc[..., -1, 3]+0.5)*(7-1)

        if hour_index is not None:
            hour_index = hour_index.long()
        if day_index is not None:
            day_index = day_index.long()

        return hour_index, day_index
        
    def forward(self, x, x_mark_enc, label=None, lable_mark=None):
        # x: (batch_size, seq_len, enc_in), hour_index: (batch_size,), day_index: (batch_size,)
        hour_index, day_index = self._index(x_mark_enc)
        # print(hour_index.shape, day_index.shape)
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x.permute(0, 2, 1)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(self.w, dim=1, norm='ortho')
        x_freq_real = x.real
        x_freq_imag = x.imag

        if self.use_hour_index and (hour_index is not None):
            emb_hour = self.emb_hour[hour_index.squeeze(-1).long() % self.emb_len_hour]
            x_freq_real = x_freq_real - emb_hour

        if self.use_day_index and (day_index is not None):
            emb_day = self.emb_day[day_index.long() % self.emb_len_day]
            x_freq_real = x_freq_real - emb_day

        x_freq_minus_emb = torch.complex(x_freq_real, x_freq_imag)
        y = x_freq_minus_emb * w
        y_real = y.real
        y_freq_imag = y.imag

        if self.use_day_index and hour_index is not None:
            y_real = y_real + emb_day
        if self.use_hour_index and day_index is not None:
            y_real = y_real + emb_hour

        y_freq = torch.complex(y_real, y_freq_imag)
        y = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        y = self.model(y).permute(0, 2, 1)

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y