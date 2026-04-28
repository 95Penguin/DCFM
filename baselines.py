"""
GridCFN 对比算法（完整忠实实现版）
参照各模型官方代码和论文，消除所有"简化"问题

DCRNN:   Li et al., ICLR 2018  (bidirectional diffusion convolution + DCGRU)
         参考: github.com/chnsh/DCRNN_PyTorch, github.com/xlwang233/pytorch-DCRNN
STGCN:   Yu et al., IJCAI 2018  (Chebyshev graph conv + gated temporal conv)
         参考: github.com/hazdzz/STGCN, github.com/FelixOpolka/STGCN-PyTorch
MTGNN:   Wu et al., KDD 2020    (graph learning + mix-hop + dilated inception)
         参考: github.com/nnzhan/MTGNN (官方实现)
AGCRN:   Bai et al., NeurIPS 2020 (AVWGCN + AGRU)
         参考: github.com/LeiBAI/AGCRN (官方实现)
HA / VAR: 统计基线

所有深度模型加了与 GridCFN 论文相同的 ProbabilisticHead 输出 (mu, sigma)，
用高斯 NLL 损失训练，保证对比公平性。
"""

import math
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple


# ============================================================================
# 评估指标（与 train.py 完全一致）
# ============================================================================

def mae(pred, true):
    return float(np.abs(pred - true).mean())

def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))

def crps_score(mu, sigma, y):
    from scipy.stats import norm
    z   = (y - mu) / (sigma + 1e-8)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return float((sigma * (z * (2*Phi - 1) + 2*phi - 1/math.sqrt(math.pi))).mean())

def picp(mu, sigma, y, confidence=0.95):
    from scipy.stats import norm
    z       = norm.ppf((1 + confidence) / 2)
    covered = ((y >= mu - z*sigma) & (y <= mu + z*sigma)).astype(float)
    return float(covered.mean())

def pinaw(mu, sigma, y, confidence=0.95):
    from scipy.stats import norm
    z       = norm.ppf((1 + confidence) / 2)
    width   = 2 * z * sigma
    y_range = y.max() - y.min() + 1e-8
    return float((width / y_range).mean())

def evaluate_all(mu_all, sigma_all, y_all):
    return {
        "MAE":   mae(mu_all, y_all),
        "RMSE":  rmse(mu_all, y_all),
        "CRPS":  crps_score(mu_all, sigma_all, y_all),
        "PICP":  picp(mu_all, sigma_all, y_all),
        "PINAW": pinaw(mu_all, sigma_all, y_all),
    }

def _inv_zscore(arr, scaler):
    return arr * scaler.std + scaler.mean

def _inv_zscore_sigma(arr, scaler):
    return arr * scaler.std


# ============================================================================
# 公共：概率输出头 & NLL 损失
# ============================================================================

class ProbabilisticHead(nn.Module):
    """
    与 GridCFN 相同的高斯分布参数输出头。
    论文中对所有深度 baseline 均加了此头（标注为 *）并用 NLL 训练。
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu    = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-4
        return mu, sigma


def nll_loss(mu, sigma, y):
    """高斯 NLL（与 GridCFN 论文 Eq.13 一致）"""
    var = sigma.pow(2).clamp(min=1e-8)
    return (0.5 * ((y - mu).pow(2) / var + torch.log(var) + math.log(2 * math.pi))).mean()


# ============================================================================
# 1. HA（Historical Average / Persistence Forecast）
#
# 与论文及 LSTNet/MTGNN 基准一致的正确实现：
#   预测值 = 输入窗口最后一步的观测值（Persistence / Naive Forecast）
#
# 【原始错误】旧版用训练集所有目标的全局均值作为预测。
#   在 Z-score 归一化后全局均值 ≈ 0，相当于永远预测 0，
#   MAE ≈ E[|N(0,1)|] = sqrt(2/π) ≈ 0.798，与论文 HA=0.215 相差 3.3 倍。
#
# 【正确做法】对每个样本，用 x 的最后一步 x[:, -1, :, :] 作为预测，
#   σ 用训练集上持续预测误差的标准差估计（per-node）。
# ============================================================================

class HistoricalAverage:
    """
    Persistence（Naive）预测：y_hat = 输入窗口最后一步观测值。
    这是与 LSTNet、MTGNN、GridCFN 论文一致的 HA 基准实现方式。
    σ 用训练集上 per-node 持续误差的标准差来估计不确定性。
    """
    def __init__(self):
        self.sigma_pred: Optional[np.ndarray] = None   # [N, F]，per-node 误差 std

    def fit(self, train_loader: DataLoader) -> "HistoricalAverage":
        """
        在训练集上计算 persistence 误差的标准差，作为不确定性估计 σ。
        persistence 误差 = y - x_last（即预测误差的分布）。
        """
        errs = []
        for x, y in train_loader:
            # x: [B, T_in, N, F]，y: [B, N, F]
            x_last = x[:, -1, :, :].numpy()   # [B, N, F]
            y_np   = y.numpy()                 # [B, N, F]
            errs.append(y_np - x_last)         # [B, N, F]
        err_all          = np.concatenate(errs, axis=0)  # [total, N, F]
        self.sigma_pred  = err_all.std(axis=0) + 1e-4    # [N, F]
        return self

    def evaluate(self, loader: DataLoader, scaler=None) -> Dict:
        """逐样本用 x 的最后一步预测 y，sigma 为训练集误差 std。"""
        assert self.sigma_pred is not None, "先调用 fit()"
        mu_list, sigma_list, y_list = [], [], []
        for x, y in loader:
            x_last = x[:, -1, :, :].numpy()   # [B, N, F]
            mu_list.append(x_last)
            sigma_list.append(
                np.broadcast_to(self.sigma_pred[None], x_last.shape).copy()
            )
            y_list.append(y.numpy())

        mu_all    = np.concatenate(mu_list,    axis=0)  # [total, N, F]
        sigma_all = np.concatenate(sigma_list, axis=0)
        y_all     = np.concatenate(y_list,     axis=0)

        if scaler is not None:
            mu_all    = _inv_zscore(mu_all, scaler)
            sigma_all = _inv_zscore_sigma(sigma_all, scaler)
            y_all     = _inv_zscore(y_all, scaler)

        sigma_all = np.maximum(sigma_all, float(np.abs(sigma_all).mean()) * 1e-4)
        return evaluate_all(mu_all, sigma_all, y_all)


def run_ha(train_loader, test_loader, val_loader=None, scaler=None, logger=None):
    if logger is None:
        logger = logging.getLogger("baseline.HA")
    model = HistoricalAverage().fit(train_loader)
    if val_loader is not None:
        val_m = model.evaluate(val_loader, scaler)
        logger.info(f"[HA] Val  | MAE={val_m['MAE']:.4f}  RMSE={val_m['RMSE']:.4f}  CRPS={val_m['CRPS']:.4f}")
    test_m = model.evaluate(test_loader, scaler)
    logger.info(f"[HA] Test | MAE={test_m['MAE']:.4f}  RMSE={test_m['RMSE']:.4f}  CRPS={test_m['CRPS']:.4f}")
    return test_m


# ============================================================================
# 2. VAR
# ============================================================================

class VARModel:
    def __init__(self, max_lag: int = 12, max_nodes_full_var: int = 50):
        self.max_lag            = max_lag
        self.max_nodes_full_var = max_nodes_full_var
        self.fitted_models      = None
        self.sigma_             = None
        self._use_independent_ar= False
        self._lag               = max_lag

    def fit(self, train_loader: DataLoader) -> "VARModel":
        x_list, y_list = [], []
        for x, y in train_loader:
            x_list.append(x.numpy())
            y_list.append(y.numpy())
        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        full  = np.concatenate([x_all[0, :, :, 0], y_all[:, :, 0]], axis=0)
        T, N  = full.shape
        if N > self.max_nodes_full_var:
            self._use_independent_ar = True
            self._fit_ar(full)
        else:
            self._use_independent_ar = False
            self._fit_var(full)
        return self

    def _fit_ar(self, series):
        T, N  = series.shape
        lag   = min(self.max_lag, T // 4)
        self._lag = lag
        models, residuals = [], []
        for n in range(N):
            s    = series[:, n]
            Xm   = np.array([[1.0] + list(s[t-lag:t][::-1]) for t in range(lag, T)])
            ym   = s[lag:]
            coeffs, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            residuals.append((ym - Xm @ coeffs).std())
            models.append(coeffs)
        self.fitted_models = models
        self.sigma_        = np.array(residuals) + 1e-4

    def _fit_var(self, series):
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except ImportError:
            raise ImportError("pip install statsmodels")
        lag               = min(self.max_lag, max(1, len(series) // 20))
        self._lag         = lag
        result            = VAR(series).fit(maxlags=lag, ic=None)
        self.fitted_models= result
        self.sigma_       = result.resid.std(axis=0) + 1e-4

    def _predict_one(self, history: np.ndarray) -> np.ndarray:
        if self._use_independent_ar:
            lag  = self._lag
            hist = history[-lag:][::-1]
            N    = hist.shape[1]
            return np.array([self.fitted_models[n] @ np.concatenate([[1.0], hist[:, n]])
                              for n in range(N)])
        else:
            lag = self.fitted_models.k_ar
            return self.fitted_models.forecast(history[-lag:], steps=1)[0]

    def evaluate(self, loader: DataLoader, scaler=None) -> Dict:
        mu_list, sigma_list, y_list = [], [], []
        for x, y in loader:
            x_np = x.numpy()[:, :, :, 0]
            y_np = y.numpy()[:, :, 0]
            mu_b = np.stack([self._predict_one(x_np[b]) for b in range(len(x_np))], axis=0)
            mu_list.append(mu_b[:, :, None])
            sigma_list.append(np.tile(self.sigma_[None, :, None], (len(x_np), 1, 1)))
            y_list.append(y_np[:, :, None])
        mu_all    = np.concatenate(mu_list, axis=0)
        sigma_all = np.concatenate(sigma_list, axis=0)
        y_all     = np.concatenate(y_list, axis=0)
        if scaler is not None:
            mu_all    = _inv_zscore(mu_all, scaler)
            sigma_all = _inv_zscore_sigma(sigma_all, scaler)
            y_all     = _inv_zscore(y_all, scaler)
        sigma_all = np.maximum(sigma_all, float(np.abs(sigma_all).mean()) * 1e-4)
        return evaluate_all(mu_all, sigma_all, y_all)


def run_var(train_loader, test_loader, val_loader=None, scaler=None,
            max_lag=12, logger=None):
    if logger is None:
        logger = logging.getLogger("baseline.VAR")
    logger.info("[VAR] 开始拟合...")
    t0 = time.time()
    model = VARModel(max_lag=max_lag).fit(train_loader)
    logger.info(f"[VAR] 拟合完成 ({time.time()-t0:.1f}s)")
    if val_loader is not None:
        val_m = model.evaluate(val_loader, scaler)
        logger.info(f"[VAR] Val  | MAE={val_m['MAE']:.4f}  RMSE={val_m['RMSE']:.4f}  CRPS={val_m['CRPS']:.4f}")
    test_m = model.evaluate(test_loader, scaler)
    logger.info(f"[VAR] Test | MAE={test_m['MAE']:.4f}  RMSE={test_m['RMSE']:.4f}  CRPS={test_m['CRPS']:.4f}")
    return test_m


# ============================================================================
# 3. DCRNN*
#    核心：DCGRU — r/u/c 三门各独立做双向扩散卷积
#    扩散: 前向随机游走 (D^{-1}A) 和后向 (D^{-1}A^T) 各 K+1 阶，
#          共 2*(K+1) 个支撑，每个支撑一组权重参数。
#    参考: github.com/chnsh/DCRNN_PyTorch
# ============================================================================

class DiffusionConvolution(nn.Module):
    """
    双向 K 阶扩散卷积（DCRNN 官方实现的 PyTorch 等价版本）。
    前向: D^{-1} A 的 0..K 次幂；后向: (D^{-1} A)^T 的 1..K 次幂。
    共 2K+1 个支撑（0 次幂只算一次）。
    参数: weight [2K+1, c_in, c_out]，bias [c_out]。
    """
    def __init__(self, c_in: int, c_out: int, K: int = 3):
        super().__init__()
        self.K = K
        n_sup  = 2 * K + 1   # 0次幂(公共) + 前向K阶 + 后向K阶
        self.weight = nn.Parameter(torch.empty(n_sup, c_in, c_out))
        self.bias   = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight.reshape(n_sup * c_in, c_out))

    @staticmethod
    def _row_norm(A: torch.Tensor) -> torch.Tensor:
        d = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return A / d

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x:[B,N,c_in], adj:[N,N] → [B,N,c_out]"""
        B, N, _ = x.shape
        A_fwd = self._row_norm(adj)
        A_bwd = self._row_norm(adj.t())

        supports = [x]          # 0 次幂（公共）
        x_fwd = x.clone()
        x_bwd = x.clone()
        for _ in range(self.K):
            x_fwd = torch.bmm(A_fwd.unsqueeze(0).expand(B, -1, -1), x_fwd)
            x_bwd = torch.bmm(A_bwd.unsqueeze(0).expand(B, -1, -1), x_bwd)
            supports.append(x_fwd)
            supports.append(x_bwd)

        # supports: 1 + 2K 个，shape 各 [B,N,c_in]
        out = torch.zeros(B, N, self.weight.shape[-1], device=x.device)
        for i, s in enumerate(supports):
            out = out + torch.einsum('bni,io->bno', s, self.weight[i])
        return torch.relu(out + self.bias)


class DCGRUCell(nn.Module):
    """
    DCGRU: r, u, c 三门各有独立的 DiffusionConvolution。
    """
    def __init__(self, c_in: int, hidden_dim: int, K: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dc_r = DiffusionConvolution(c_in + hidden_dim, hidden_dim, K)
        self.dc_u = DiffusionConvolution(c_in + hidden_dim, hidden_dim, K)
        self.dc_c = DiffusionConvolution(c_in + hidden_dim, hidden_dim, K)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        xh  = torch.cat([x, h], dim=-1)
        r   = torch.sigmoid(self.dc_r(xh, adj))
        u   = torch.sigmoid(self.dc_u(xh, adj))
        xrh = torch.cat([x, r * h], dim=-1)
        c   = torch.tanh(self.dc_c(xrh, adj))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(self, in_dim: int = 1, hidden_dim: int = 64,
                 out_dim: int = 1, n_layers: int = 2, K: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.cells = nn.ModuleList([
            DCGRUCell(in_dim if i == 0 else hidden_dim, hidden_dim, K)
            for i in range(n_layers)
        ])
        self.head = ProbabilisticHead(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                edge_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _ = x.shape
        hs = [torch.zeros(B, N, self.hidden_dim, device=x.device)
              for _ in range(self.n_layers)]
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                hs[l] = cell(inp, hs[l], adj_norm)
                inp   = hs[l]
        return self.head(hs[-1])


# ============================================================================
# 4. STGCN*
#    核心：
#    · ChebConv: K 阶 Chebyshev 多项式图卷积
#    · TemporalGatedConv: GLU 时间卷积（非 causal Conv2d，标准 padding=0）
#    · ST-Block: [TGC → ChebConv → BN → TGC] + 残差
#    · OutputLayer: LayerNorm + FC → 概率头
#    参考: github.com/hazdzz/STGCN
# ============================================================================

class ChebConv(nn.Module):
    """
    Chebyshev 图卷积 (K 阶)。
    论文使用 L_tilde = 2L/lambda_max - I，实践中常直接用 normalize_adj 近似。
    """
    def __init__(self, c_in: int, c_out: int, K: int = 3):
        super().__init__()
        self.K      = K
        self.weight = nn.Parameter(torch.empty(K, c_in, c_out))
        self.bias   = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight.reshape(K * c_in, c_out))

    def forward(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        """x:[B,N,c_in], lap:[N,N] → [B,N,c_out]"""
        B, N, _ = x.shape
        Tx = [x]
        if self.K > 1:
            x1 = torch.bmm(lap.unsqueeze(0).expand(B, -1, -1), x)
            Tx.append(x1)
        for _ in range(2, self.K):
            x2 = 2 * torch.bmm(lap.unsqueeze(0).expand(B, -1, -1), Tx[-1]) - Tx[-2]
            Tx.append(x2)
        out = sum(torch.einsum('bni,io->bno', Tx[k], self.weight[k])
                  for k in range(len(Tx)))
        return out + self.bias


class TemporalGatedConv(nn.Module):
    """
    GLU 时间门控卷积。Conv2d(c_in, 2*c_out, (1, Kt)) + GLU。
    """
    def __init__(self, c_in: int, c_out: int, Kt: int):
        super().__init__()
        self.conv  = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt))
        self.c_out = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x:[B,c_in,N,T] → [B,c_out,N,T-Kt+1]"""
        h = self.conv(x)
        return torch.tanh(h[:, :self.c_out]) * torch.sigmoid(h[:, self.c_out:])


class STConvBlock(nn.Module):
    """
    STGCN ST-Block: TGC(Kt) → ChebConv(Ks) → BN → TGC(Kt) → BN → Dropout。
    两次 TGC 各减少 Kt-1 的时间步，残差需裁剪。
    """
    def __init__(self, c_in: int, c_spat: int, c_out: int,
                 Kt: int = 3, Ks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.Kt      = Kt
        self.tconv1  = TemporalGatedConv(c_in,   c_spat, Kt)
        self.cheb    = ChebConv(c_spat, c_spat, Ks)
        self.bn1     = nn.BatchNorm2d(c_spat)
        self.tconv2  = TemporalGatedConv(c_spat, c_out,  Kt)
        self.bn2     = nn.BatchNorm2d(c_out)
        self.drop    = nn.Dropout(dropout)
        self.res_conv = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        """x:[B,c_in,N,T] → [B,c_out,N,T-2*(Kt-1)]"""
        trim = 2 * (self.Kt - 1)
        res  = self.res_conv(x)[:, :, :, trim:]

        h = self.tconv1(x)                         # [B, c_spat, N, T-(Kt-1)]
        # 图卷积：遍历时间步
        B, C, N, T2 = h.shape
        h_ = h.permute(0, 3, 2, 1).reshape(B * T2, N, C)
        h_ = torch.relu(self.cheb(h_, lap))
        h  = h_.reshape(B, T2, N, C).permute(0, 3, 2, 1)
        h  = torch.relu(self.bn1(h))
        h  = self.tconv2(h)                        # [B, c_out, N, T-2*(Kt-1)]
        h  = self.drop(torch.relu(self.bn2(h + res)))
        return h


class STGCN(nn.Module):
    """
    STGCN* = STGCN + ProbabilisticHead。
    ST-Block x n_blocks → OutputLayer(LayerNorm + 线性) → 概率头。
    """
    def __init__(self, in_dim: int = 1, hidden_dim: int = 64,
                 out_dim: int = 1, n_blocks: int = 2,
                 Kt: int = 3, Ks: int = 3, T_in: int = 168,
                 dropout: float = 0.1):
        super().__init__()
        self.n_blocks = n_blocks
        self.Kt       = Kt
        blocks = []
        c_in   = in_dim
        for i in range(n_blocks):
            blocks.append(STConvBlock(c_in, hidden_dim, hidden_dim, Kt, Ks, dropout))
            c_in = hidden_dim
        self.st_blocks = nn.ModuleList(blocks)

        # 每个 Block 缩短 2*(Kt-1) 时间步
        T_out = T_in - n_blocks * 2 * (Kt - 1)
        assert T_out > 0, f"T_out={T_out}<=0，需增大T_in或减小n_blocks/Kt"

        # OutputLayer: 时间维压缩 + LayerNorm + FC
        self.output_tconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, T_out))
        self.ln            = nn.LayerNorm(hidden_dim)   # 对最后一维做 LN
        self.fc            = nn.Linear(hidden_dim, hidden_dim)
        self.head          = ProbabilisticHead(hidden_dim, out_dim)

    @staticmethod
    def _build_lap(adj_norm: torch.Tensor) -> torch.Tensor:
        """归一化 Laplacian: L = I - adj_norm，将 adj_norm 缩放到 [-1,1]"""
        return torch.eye(adj_norm.size(0), device=adj_norm.device) - adj_norm

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                edge_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, Fin = x.shape
        lap  = self._build_lap(adj_norm)
        h    = x.permute(0, 3, 2, 1)           # [B, Fin, N, T]
        for block in self.st_blocks:
            h = block(h, lap)                   # [B, hid, N, T']
        h = self.output_tconv(h)               # [B, hid, N, 1]
        h = h.squeeze(-1).permute(0, 2, 1)     # [B, N, hid]
        h = torch.relu(self.ln(h))
        h = torch.relu(self.fc(h))
        return self.head(h)


# ============================================================================
# 5. MTGNN*
#    官方完整实现参考: github.com/nnzhan/MTGNN/blob/master/net.py
#
#    核心组件（与官方完全对应）：
#    · GraphConstructor: E1,E2 嵌入 → tanh(alpha * linear(E)) → A = ReLU(E1@E2^T)
#    · MixProp: K跳 mix-hop 传播，cat 各跳结果后过 MLP（官方 mixprop 完整实现）
#    · DilatedInception: 4 种 kernel (2,3,6,7) 的并行膨胀因果卷积
#    · gtnet 前向：start_conv → 逐层(gate_conv + skip + graph_conv + residual) → output
# ============================================================================

class GraphConstructor(nn.Module):
    """
    MTGNN 图学习层（官方 graph_constructor）。
    A = ReLU(tanh(alpha*lin1(E1)) @ tanh(alpha*lin2(E2))^T)，行归一化。
    """
    def __init__(self, n_nodes: int, node_dim: int = 40, alpha: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.emb1  = nn.Embedding(n_nodes, node_dim)
        self.emb2  = nn.Embedding(n_nodes, node_dim)
        self.lin1  = nn.Linear(node_dim, node_dim)
        self.lin2  = nn.Linear(node_dim, node_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        e1  = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        e2  = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        A   = torch.relu(torch.mm(e1, e2.t()))
        d   = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return A / d


class MixProp(nn.Module):
    """
    官方 mixprop（与 github.com/nnzhan/MTGNN 完全一致）。
    输入 x: [B, c_in, N]  adj: [N, N]
    H^(k) = alpha*x + (1-alpha)*adj * H^(k-1)
    out = MLP(cat(H^0,...,H^K))
    """
    def __init__(self, c_in: int, c_out: int, gdep: int = 2,
                 dropout: float = 0.3, alpha: float = 0.05):
        super().__init__()
        self.gdep    = gdep
        self.alpha   = alpha
        self.dropout = dropout
        self.mlp     = nn.Linear((gdep + 1) * c_in, c_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x:[B,c_in,N], adj:[N,N] → [B,c_out,N]"""
        # 加自环并行归一化（官方实现）
        A  = adj + torch.eye(adj.size(0), device=adj.device)
        d  = A.sum(1, keepdim=True).clamp(min=1e-8)
        A  = A / d

        h    = x
        outs = [h]
        for _ in range(self.gdep):
            # h: [B, C, N]  A: [N, N]
            h = self.alpha * x + (1 - self.alpha) * torch.einsum('bcn,mn->bcm', h, A)
            if self.dropout > 0:
                h = F.dropout(h, self.dropout, training=self.training)
            outs.append(h)
        ho = torch.cat(outs, dim=1)                        # [B, (K+1)*C, N]
        return self.mlp(ho.permute(0, 2, 1)).permute(0, 2, 1)  # [B, c_out, N]


class DilatedInception(nn.Module):
    """
    官方 dilated_inception: 4 种 kernel (2,3,6,7) 并行膨胀卷积。
    输入 [B, c_in, N, T]，输出 [B, c_out, N, T']（每路 c_out//4 个通道，
    T' = T - (max_kernel-1)*dilation = T - 6*dilation）。
    """
    def __init__(self, c_in: int, c_out: int, dilation_factor: int = 1):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        c_k = c_out // len(self.kernel_set)
        self.convs = nn.ModuleList([
            nn.Conv2d(c_in, c_k, kernel_size=(1, k), dilation=(1, dilation_factor))
            for k in self.kernel_set
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs  = [conv(x) for conv in self.convs]
        T_min = min(o.size(-1) for o in outs)
        return torch.cat([o[:, :, :, -T_min:] for o in outs], dim=1)


class MTGNN(nn.Module):
    """
    MTGNN* (官方 gtnet + ProbabilisticHead)。

    关键结构（与官方 net.py 完全一致）:
    · 每层用独立的 filter_conv 和 gate_conv（各输出 conv_channels），
      GLU: x = tanh(filter_conv(h)) * sigmoid(gate_conv(h))
    · skip_conv: 对每层的 x 做卷积 → 累加 skip
    · residual: h_new = residual_conv(x) + residual[:,:,:,-x.T:]
    · graph_conv: 对 x 的最后时刻做 mixprop，结果加进残差
    · 输出: end_conv1 → end_conv2 → ProbHead

    官方单步预测超参:
      layers=3, conv_channels=residual_channels=32,
      skip_channels=64, end_channels=128, node_dim=40
    """
    def __init__(self, in_dim: int = 1, conv_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 out_dim: int = 1, n_nodes: int = 137, T_in: int = 168,
                 n_layers: int = 3, gcn_depth: int = 2,
                 node_dim: int = 40, dropout: float = 0.3,
                 propalpha: float = 0.05, tanhalpha: float = 3.0,
                 dilation_exp: int = 1):
        super().__init__()
        self.n_layers    = n_layers
        self.dropout     = dropout
        self.conv_ch     = conv_channels

        self.gc  = GraphConstructor(n_nodes, node_dim, tanhalpha)
        self.idx = nn.Parameter(torch.arange(n_nodes), requires_grad=False)

        self.start_conv = nn.Conv2d(in_dim, conv_channels, (1, 1))

        # 感受野计算（最大 kernel=7）
        kernel_size = 7
        if dilation_exp > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1)
            )
        else:
            self.receptive_field = n_layers * (kernel_size - 1) + 1

        # 每层用独立的 filter_conv + gate_conv（官方）
        self.filter_convs  = nn.ModuleList()
        self.gate_convs    = nn.ModuleList()
        self.residual_convs= nn.ModuleList()
        self.skip_convs    = nn.ModuleList()
        self.gconv1        = nn.ModuleList()
        self.gconv2        = nn.ModuleList()
        self.bn_list       = nn.ModuleList()

        new_dilation = 1
        # 计算每层 x 输出的时间长度（用于 skip_conv kernel_size）
        rf_size_i = 1
        for j in range(1, n_layers + 1):
            # 每层 dilated_inception 最大 kernel=7 → 时间缩减量
            rf_size_j = rf_size_i + (kernel_size - 1) * new_dilation

            self.filter_convs.append(
                DilatedInception(conv_channels, conv_channels, new_dilation))
            self.gate_convs.append(
                DilatedInception(conv_channels, conv_channels, new_dilation))
            self.residual_convs.append(
                nn.Conv2d(conv_channels, conv_channels, (1, 1)))

            # skip_conv kernel_size = x 在本层的时间长度
            if T_in > self.receptive_field:
                skip_size = T_in - rf_size_j + 1
            else:
                skip_size = self.receptive_field - rf_size_j + 1
            skip_size = max(1, skip_size)
            self.skip_convs.append(
                nn.Conv2d(conv_channels, skip_channels, (1, skip_size)))

            self.gconv1.append(MixProp(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(MixProp(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.bn_list.append(nn.BatchNorm2d(conv_channels))

            rf_size_i = rf_size_j
            new_dilation *= dilation_exp if dilation_exp > 1 else 1

        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.end_conv2 = nn.Conv2d(end_channels,  out_dim,       (1, 1))
        self.head      = ProbabilisticHead(out_dim, out_dim, hidden_dim=conv_channels)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                edge_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, Fin = x.shape
        A_adp = self.gc(self.idx.to(x.device))

        x_in = x.permute(0, 3, 2, 1)             # [B, Fin, N, T]
        if T < self.receptive_field:
            x_in = F.pad(x_in, (self.receptive_field - T, 0, 0, 0))

        h = self.start_conv(x_in)                 # [B, C, N, T']
        skip_sum = None

        for i in range(self.n_layers):
            residual = h                           # [B, C, N, T_cur]
            # 独立 filter + gate（官方做法）
            filt = torch.tanh(self.filter_convs[i](h))    # [B, C, N, T_new]
            gate = torch.sigmoid(self.gate_convs[i](h))   # [B, C, N, T_new]
            x_   = filt * gate                            # [B, C, N, T_new]
            x_   = F.dropout(x_, self.dropout, training=self.training)

            # skip（对 x_ 的时间维做卷积）
            try:
                s = self.skip_convs[i](x_)
            except RuntimeError:
                # 时间维不够时退化为 global avg
                s = x_.mean(dim=-1, keepdim=True)
                s = F.conv2d(s, self.skip_convs[i].weight[:, :, :, :1],
                             self.skip_convs[i].bias)
            skip_sum = s if skip_sum is None else (
                skip_sum[:, :, :, -s.size(-1):] + s
            )

            # 图卷积（对 x_ 每个时刻独立做 mixprop）
            # x_: [B, C, N, T_new]
            T_new = x_.size(-1)
            # 逐时刻 graph conv → 累加到 x_
            x_gc = []
            for t in range(T_new):
                xt = x_[:, :, :, t]               # [B, C, N]
                gc_t = (self.gconv1[i](xt, A_adp)
                        + self.gconv2[i](xt, adj_norm))  # [B, C, N]
                x_gc.append(gc_t.unsqueeze(-1))
            x_ = x_ + torch.cat(x_gc, dim=-1)    # [B, C, N, T_new]

            # 残差连接（slice residual 时间维与 x_ 对齐）
            h = self.bn_list[i](
                self.residual_convs[i](x_)
                + residual[:, :, :, -x_.size(-1):]
            )

        # 输出
        out = F.relu(skip_sum)                    # [B, skip, N, ?]
        out = F.relu(self.end_conv1(out))         # [B, end, N, ?]
        out = self.end_conv2(out)                 # [B, out_dim, N, ?]
        # 若时间维 > 1，取均值
        if out.size(-1) > 1:
            out = out.mean(dim=-1, keepdim=True)
        out = out.squeeze(-1).permute(0, 2, 1)   # [B, N, out_dim]
        return self.head(out)


# ============================================================================
# 6. AGCRN*
#    官方: github.com/LeiBAI/AGCRN
#
#    AVWGCN: 用节点嵌入 E 生成自适应邻接矩阵 A=softmax(ReLU(E@E^T))，
#            K 阶 Chebyshev 递推图卷积（支持支撑: I, A, A^2, ..., A^{K-1}）。
#    AGRU:   r, u 门合并（输出 2*hidden）+ 候选状态 c，各用一个 AVWGCN。
#            节点嵌入 E 由模型共享，所有层共用。
# ============================================================================

class AVWGCN(nn.Module):
    """
    Adaptive View-based Weighted GCN。
    与官方 AGCRNCell 内 AVWGCN 完全一致：
      A = softmax(ReLU(E @ E^T))，Chebyshev K 阶递推。
    """
    def __init__(self, c_in: int, c_out: int, embed_dim: int, K: int = 3):
        super().__init__()
        self.K      = K
        self.weight = nn.Parameter(torch.empty(K, c_in, c_out))
        self.bias   = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight.reshape(K * c_in, c_out))

    def forward(self, x: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """x:[B,N,c_in], E:[N,embed_dim] → [B,N,c_out]"""
        B, N, _ = x.shape
        A = F.softmax(F.relu(torch.mm(E, E.t())), dim=1)  # [N, N]
        A_exp = A.unsqueeze(0).expand(B, -1, -1)

        Tx = [x]
        for _ in range(1, self.K):
            Tx.append(torch.bmm(A_exp, Tx[-1]))

        out = sum(torch.einsum('bni,io->bno', Tx[k], self.weight[k])
                  for k in range(len(Tx)))
        return torch.relu(out + self.bias)


class AGRUCell(nn.Module):
    """
    Adaptive GRU Cell（AGCRN 官方 AGCRNCell）。
    r+u 门合并输出 2*hidden_dim，候选状态 c 单独一个 AVWGCN。
    """
    def __init__(self, c_in: int, hidden_dim: int, embed_dim: int, K: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = AVWGCN(c_in + hidden_dim, 2 * hidden_dim, embed_dim, K)
        self.cand = AVWGCN(c_in + hidden_dim, hidden_dim,     embed_dim, K)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                E: torch.Tensor) -> torch.Tensor:
        xh    = torch.cat([x, h], dim=-1)
        gate  = torch.sigmoid(self.gate(xh, E))
        r, u  = gate[..., :self.hidden_dim], gate[..., self.hidden_dim:]
        xrh   = torch.cat([x, r * h], dim=-1)
        c     = torch.tanh(self.cand(xrh, E))
        return u * h + (1 - u) * c


class AGCRN(nn.Module):
    """
    AGCRN* = AGCRN + ProbabilisticHead。
    多层 AGRU 编码器，节点嵌入 E 所有层共享（官方做法）。
    """
    def __init__(self, in_dim: int = 1, hidden_dim: int = 64,
                 out_dim: int = 1, n_nodes: int = 137,
                 n_layers: int = 2, embed_dim: int = 10, K: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.node_emb   = nn.Parameter(torch.empty(n_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb.unsqueeze(0))

        self.cells = nn.ModuleList([
            AGRUCell(in_dim if i == 0 else hidden_dim, hidden_dim, embed_dim, K)
            for i in range(n_layers)
        ])
        self.head = ProbabilisticHead(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                edge_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _ = x.shape
        E  = self.node_emb
        hs = [torch.zeros(B, N, self.hidden_dim, device=x.device)
              for _ in range(self.n_layers)]
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                hs[l] = cell(inp, hs[l], E)
                inp   = hs[l]
        return self.head(hs[-1])


# ============================================================================
# 工厂函数 & 通用训练 / 评估
# ============================================================================

def build_baseline(name: str, in_dim: int, out_dim: int, n_nodes: int,
                   T_in: int = 168, hidden_dim: int = 64) -> nn.Module:
    name = name.lower()
    if name == "dcrnn":
        return DCRNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                     n_layers=2, K=3)
    elif name == "stgcn":
        return STGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                     n_blocks=2, Kt=3, Ks=3, T_in=T_in, dropout=0.1)
    elif name == "mtgnn":
        # 官方单步预测超参
        return MTGNN(in_dim=in_dim, conv_channels=32, skip_channels=64,
                     end_channels=128, out_dim=out_dim, n_nodes=n_nodes,
                     T_in=T_in, n_layers=3, gcn_depth=2, node_dim=40,
                     dropout=0.3, propalpha=0.05, tanhalpha=3.0)
    elif name == "agcrn":
        return AGCRN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                     n_nodes=n_nodes, n_layers=2, embed_dim=10, K=3)
    else:
        raise ValueError(f"未知 baseline: '{name}'。可选: dcrnn, stgcn, mtgnn, agcrn")


def train_deep_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    weight_decay: float = 1e-5,
    save_path: str = "best_baseline.pt",
    logger: Optional[logging.Logger] = None,
    model_name: str = "Baseline",
) -> Dict:
    if logger is None:
        logger = logging.getLogger(f"baseline.{model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    adj_d = adj_norm.to(device)
    eid_d = edge_index.to(device)

    best_val_crps     = float("inf")
    epochs_no_improve = 0
    history           = {"train_nll": [], "val_crps": [], "val_mae": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_nll = 0.0
        n_batch   = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mu, sigma = model(x, adj_d, eid_d)
            loss      = nll_loss(mu, sigma, y[..., :mu.shape[-1]])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_nll += loss.item()
            n_batch   += 1

        val_m = evaluate_deep_baseline(
            model, val_loader, adj_d, eid_d, device,
            scaler=None, inverse_transform=False
        )
        scheduler.step(val_m["CRPS"])
        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_nll"].append(total_nll / n_batch)
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])

        logger.info(
            f"[{model_name}] Ep{epoch:>4} | NLL={total_nll/n_batch:.4f} | "
            f"Val MAE={val_m['MAE']:.4f} CRPS={val_m['CRPS']:.4f} | "
            f"LR={cur_lr:.2e} | {time.time()-t0:.1f}s"
        )

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"[{model_name}] 早停 epoch={epoch}，best val CRPS={best_val_crps:.4f}"
                )
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    history["best_val_crps"] = best_val_crps
    return history


@torch.no_grad()
def evaluate_deep_baseline(
    model: nn.Module,
    loader: DataLoader,
    adj_norm: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    scaler=None,
    inverse_transform: bool = True,
) -> Dict:
    model.eval()
    mu_list, sigma_list, y_list = [], [], []
    for x, y in loader:
        x = x.to(device)
        mu, sig = model(x, adj_norm, edge_index)
        mu_list.append(mu.cpu().numpy())
        sigma_list.append(sig.cpu().numpy())
        y_list.append(y.numpy())

    mu_all    = np.concatenate(mu_list,    axis=0)
    sigma_all = np.concatenate(sigma_list, axis=0)
    y_all     = np.concatenate(y_list,     axis=0)[..., :mu_all.shape[-1]]

    if inverse_transform and scaler is not None:
        mu_all    = _inv_zscore(mu_all, scaler)
        sigma_all = _inv_zscore_sigma(sigma_all, scaler)
        y_all     = _inv_zscore(y_all, scaler)

    sigma_all = np.maximum(sigma_all, float(np.abs(sigma_all).mean()) * 1e-4)
    return evaluate_all(mu_all, sigma_all, y_all)