import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MultiScaleGradientAnalyzer:
    def __init__(self, model):
        """
        初始化分析器
        :param model: 待分析的模型 (必须在 train 模式下，以保留计算图)
        """
        self.model = model
        # 冻结模型防止参数更新，但保留计算图以便求导
        self.model.eval() 
        for p in self.model.parameters():
            p.requires_grad = True

    def get_target_params(self):
        """
        获取用于计算梯度的核心参数。
        """
        # 默认返回所有需要梯度的参数
        return [p for p in self.model.parameters() if p.requires_grad]

    def _compute_grad_vector(self, loss, params):
        """辅助函数：计算梯度并展平为向量"""
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        # 处理可能的 None 梯度 (针对 DLinear 这种部分参数不参与计算的情况)
        valid_grads = []
        for g, p in zip(grads, params):
            if g is not None:
                valid_grads.append(g.view(-1))
            else:
                valid_grads.append(torch.zeros_like(p).view(-1))
        return torch.cat(valid_grads)

    def analyze_segments(self, x, y_true):
        """
        Analyze a single batch and return raw metrics.
        Returns:
            - norm_ratios: dict {segment: ratio}
            - pairwise_matrix: np.array (5x5)
        """
        self.model.zero_grad()
        params = self.get_target_params()
        
        # Segments Definition
        segments = {
            "96": slice(0, 96),
            "192": slice(96, 192),
            "336": slice(192, 336),
            "720": slice(336, 720),
            "All": slice(0, 720)
        }
        seg_keys = list(segments.keys()) # ['96', '192', '336', '720', 'All']
        
        # 1. Forward Pass
        if isinstance(x, (list, tuple)):
            y_pred = self.model(*x)
        else:
            y_pred = self.model(x, None, None, None) # Adjust based on model input signature
            
        if isinstance(y_pred, tuple): y_pred = y_pred[0]
        
        segment_grads = {}
        segment_norms = {}
        
        # 2. Compute Gradients per Segment
        for name, slc in segments.items():
            loss_seg = F.mse_loss(y_pred[:, slc, :], y_true[:, slc, :])
            grad_vec = self._compute_grad_vector(loss_seg, params)
            segment_grads[name] = grad_vec
            segment_norms[name] = grad_vec.norm(2).item()
            
        # 3. Compute Metrics
        norm_total = segment_norms["All"]
        norm_ratios = {}
        
        # A. Norm Ratios (vs Total) - Skip 'All' for ratio dict as it is 1.0
        for name in ["96", "192", "336", "720"]: 
            norm_ratios[name] = segment_norms[name] / (norm_total + 1e-9)
            
        # B. Pairwise Similarity Matrix (5x5 including All)
        n_segs = len(seg_keys)
        pairwise_matrix = np.zeros((n_segs, n_segs))
        
        for i, k1 in enumerate(seg_keys):
            for j, k2 in enumerate(seg_keys):
                if i == j:
                    pairwise_matrix[i, j] = 1.0
                elif i > j:
                    pairwise_matrix[i, j] = pairwise_matrix[j, i] # Symmetric
                else:
                    g1 = segment_grads[k1]
                    g2 = segment_grads[k2]
                    sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                    pairwise_matrix[i, j] = sim
                    
        return norm_ratios, pairwise_matrix

    # def analyze_segments(self, x, y_true):
    #     """
    #     执行任务:
    #     1. Directional Conflict (分段 vs Total, 分段 vs 分段)
    #     2. Signal Masking (范数比例)
    #     分段：1-96, 97-192, 193-336, 337-720
    #     """
    #     self.model.zero_grad()
    #     params = self.get_target_params()
        
    #     # 定义尺度切片
    #     segments = {
    #         "96": slice(0, 96),
    #         "192": slice(96, 192),
    #         "336": slice(192, 336),
    #         "720": slice(336, 720),
    #         "All": slice(0, 720)
    #     }
        
    #     # 1. Forward Pass
    #     if isinstance(x, (list, tuple)): # 修正语法: isinstance第二个参数应为tuple
    #         y_pred = self.model(*x)
    #     else:
    #         # 这里的参数调用视具体模型而定，保持你原有的逻辑
    #         y_pred = self.model(x, None, None, None) 
        
    #     if isinstance(y_pred, tuple): y_pred = y_pred[0]
        
    #     segment_grads = {}
    #     segment_norms = {}
        
    #     # 2. 计算各分段的梯度
    #     for name, slc in segments.items():
    #         loss_seg = F.mse_loss(y_pred[:, slc, :], y_true[:, slc, :])
    #         grad_vec = self._compute_grad_vector(loss_seg, params)
    #         segment_grads[name] = grad_vec
    #         segment_norms[name] = grad_vec.norm(2).item()
            
    #     # 3. 计算总长度 (0-720) 的梯度 (作为 Anchor)
    #     loss_total = F.mse_loss(y_pred[:, :720, :], y_true[:, :720, :])
    #     grad_total = self._compute_grad_vector(loss_total, params)
    #     norm_total = grad_total.norm(2).item()
        
    #     # 4. 汇总指标 (Part A: Segment vs Total)
    #     print(f"\n{'Segment':<10} | {'Cos Sim (vs Total)':<20} | {'Norm Ratio (vs Total)':<20} | {'Norm Value':<15}")
    #     print("-" * 75)
        
    #     metrics_data = []
    #     seg_keys = list(segments.keys()) # 保持顺序: ['96', '192', '336', '720']
        
    #     for name in seg_keys:
    #         g_seg = segment_grads[name]
            
    #         # --- Directional Conflict (vs Total) ---
    #         cos_sim = F.cosine_similarity(g_seg.unsqueeze(0), grad_total.unsqueeze(0)).item()
            
    #         # --- Signal Masking ---
    #         norm_ratio = segment_norms[name] / (norm_total + 1e-9)
            
    #         metrics_data.append({
    #             "Segment": name,
    #             "Cosine Sim (vs Total)": cos_sim,
    #             "Norm Ratio": norm_ratio,
    #             "Norm": segment_norms[name]
    #         })
            
    #         print(f"{name:<10} | {cos_sim:<20.4f} | {norm_ratio:<20.4f} | {segment_norms[name]:<15.4f}")
            
    #     # 5. 计算相互之间的相似度 (Part B: Pairwise Similarity)
    #     n_segs = len(seg_keys)
    #     pairwise_matrix = np.zeros((n_segs, n_segs))
        
    #     print(f"\n{'Pairwise Similarity Matrix':^50}")
    #     print("-" * 50)
        
    #     for i, k1 in enumerate(seg_keys):
    #         for j, k2 in enumerate(seg_keys):
    #             if i == j:
    #                 pairwise_matrix[i, j] = 1.0
    #             elif i > j: # 矩阵是对称的，直接复制
    #                 pairwise_matrix[i, j] = pairwise_matrix[j, i]
    #             else:
    #                 g1 = segment_grads[k1]
    #                 g2 = segment_grads[k2]
    #                 sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
    #                 pairwise_matrix[i, j] = sim
        
    #     # 创建 DataFrame 并打印
    #     df_pairwise = pd.DataFrame(pairwise_matrix, index=seg_keys, columns=seg_keys)
    #     print(df_pairwise.round(4))
        
    #     return pd.DataFrame(metrics_data), df_pairwise



    def analyze_snr_trend(self, x, y_true, step_stride=1):
        """
        执行任务 3 (SNR & Variance)
        计算 1 到 720 每个时间步的梯度方差和 SNR
        """
        self.model.zero_grad()
        params = self.get_target_params()
        batch_size = x.shape[0]
        
        # 采样时间步，避免计算太慢
        time_steps = list(range(0, 720, step_stride))
        snr_stats = []
        
        print(f"\nComputing SNR for {len(time_steps)} steps (stride={step_stride})...")
        
        # 预计算：为了计算逐样本梯度，我们需要 Per-sample Loop
        # (这是计算 Gradient Variance 最通用的方法，虽然慢但稳)
        
        # 这里的策略是：针对每个选定的时间步 t
        # 我们计算 Loss_t = MSE(y_pred[:, t, :], y_true[:, t, :])
        # 然后求 gradients 的均值和方差
        
        for t in tqdm(time_steps):
            # 存储该时间步下，Batch 中每个样本的梯度向量
            batch_grads = []
            
            for i in range(batch_size):
                # 取单个样本
                x_i = x[i:i+1]
                y_i = y_true[i:i+1]
                
                # Forward (必需每次做以保持 graph 独立，或使用 torch.func)
                pred_i = self.model(x_i, None, None, None)
                if isinstance(pred_i, tuple): pred_i = pred_i[0]
                
                # 计算单步 Loss
                loss_t = F.mse_loss(pred_i[:, t:t+1, :], y_i[:, t:t+1, :])
                
                # 计算梯度
                g_vec = self._compute_grad_vector(loss_t, params)
                batch_grads.append(g_vec)
            
            # 堆叠所有样本的梯度: [Batch, Num_Params]
            # 注意：如果参数量极大，这里会爆显存。建议只对 Encoder 最后一层做。
            grad_matrix = torch.stack(batch_grads)
            
            # --- 任务 3: SNR 计算 ---
            # 1. Mean Gradient (信号强度)
            mean_grad = torch.mean(grad_matrix, dim=0)
            signal_power = torch.norm(mean_grad).item() ** 2
            
            # 2. Total Variance (噪声强度) - Sum of variances of each parameter
            variance = torch.var(grad_matrix, dim=0, unbiased=True).sum().item()
            
            # 3. SNR
            snr = signal_power / (variance + 1e-9)
            
            snr_stats.append({
                "Time_Step": t + 1,
                "Signal_Power": signal_power,
                "Noise_Variance": variance,
                "SNR": snr
            })
            
            if (t+1) % 100 == 1:
                print(f"Step {t+1}: SNR = {snr:.4f}")
                
        return pd.DataFrame(snr_stats)

# --- 辅助绘图函数 (更新版) ---
def plot_analysis_results(df_metrics, df_pairwise, model_name, dataset_name):
    """
    绘制完整的分析图表：
    1. Signal Masking (Bar plot)
    2. Sim vs Total (Bar plot)
    3. Pairwise Sim (Heatmap)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 1. Signal Masking
    sns.barplot(data=df_metrics, x='Segment', y='Norm Ratio', ax=axes[0], palette='viridis')
    axes[0].set_title("Signal Masking: Norm Ratio (vs Total)")
    for i, v in enumerate(df_metrics['Norm Ratio']):
        axes[0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    # 2. Cosine Sim vs Total
    sns.barplot(data=df_metrics, x='Segment', y='Cosine Sim (vs Total)', ax=axes[1], palette='coolwarm')
    axes[1].set_title("Conflict: Similarity with Total Gradient")
    axes[1].set_ylim(-0.2, 1.1)
    
    # 3. Pairwise Similarity Heatmap (新增)
    sns.heatmap(df_pairwise, annot=True, cmap="RdBu_r", vmin=-0.5, vmax=1.0, ax=axes[2], fmt=".2f")
    axes[2].set_title("Internal Conflict: Pairwise Gradient Similarity")
    
    plt.tight_layout()
    # Correct
    plt.savefig('/root/autodl-tmp/TSL/draw/' + model_name + '_' + dataset_name + '.png')
    plt.show()

# --- 使用示例 ---
# 假设 model, x_batch, y_batch 已经准备好
# analyzer = MultiScaleGradientAnalyzer(model)
# df_segs = analyzer.analyze_segments(x_batch, y_batch)
# df_snr = analyzer.analyze_snr_trend(x_batch, y_batch, step_stride=24) # 步长设大点跑得快
# plot_analysis_results(df_segs, df_snr)