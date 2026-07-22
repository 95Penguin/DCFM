# cfm_alternatives.py
# 本文件位于项目根目录的 ablation/ 目录下，与 ablation_model.py 并列。
"""
CFM 替代输出头，用于 noCFM 消融实验。

本文件提供两个与 CFMVectorField 接口对齐的替代输出头：

  GaussianHead   : 参数化高斯输出头（变体 B）
    - forward(He_prime, Hs_prime) -> (mu, logvar)  [B, N, cfm_dim]
    - 训练用 NLL loss
    - 推断时从 N(mu, sigma²) 采 S 个样本，保持与 CFM sample() 相同的输出格式

  DeterministicHead : 确定性点预测 MLP 头（变体 A）
    - forward(He_prime, Hs_prime) -> mu  [B, N, cfm_dim]
    - 训练用 MSE loss
    - 推断时把同一个 mu 复制 S 份，保持输出格式与 CFM 一致（宽度为 0 的分布）

接口设计原则：
  - 两个头都暴露 loss(He_prime, Hs_prime, y_target) 和
    sample(He_prime, Hs_prime, n_samples, ...) 方法，
    与 DCFM.cfm_loss / DCFM.sample 签名对齐，
    使 ablation_train.py 不需要为它们写专门的训练循环。
  - sample() 的返回值 shape 与 DCFM.sample() 完全一致：
    [S, B, N, T_out, feat_dim]，evaluate() 可以直接复用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseAlternativeHead(nn.Module):
    """公共基类：持有 T_out / feat_dim / cfm_dim，并提供 _build_net 工具。"""

    def __init__(self, env_dim: int, stoch_dim: int, hidden_dim: int,
                 T_out: int, feat_dim: int, dropout: float = 0.1):
        super().__init__()
        self.T_out    = T_out
        self.feat_dim = feat_dim
        self.cfm_dim  = T_out * feat_dim
        self._env_dim   = env_dim
        self._stoch_dim = stoch_dim
        self._hidden    = hidden_dim

    def _build_net(self, out_dim: int, dropout: float) -> nn.Sequential:
        """两层 MLP，输入 = He_prime concat Hs_prime。"""
        in_dim = self._env_dim + self._stoch_dim
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self._hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self._hidden, self._hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self._hidden, out_dim),
        )

    def _concat(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor) -> torch.Tensor:
        """[B,N,env_dim] + [B,N,stoch_dim] -> [B,N,env_dim+stoch_dim]"""
        return torch.cat([He_prime, Hs_prime], dim=-1)


# ---------------------------------------------------------------------------
# 变体 A：确定性点预测头
# ---------------------------------------------------------------------------

class DeterministicHead(_BaseAlternativeHead):
    """
    用 MLP 直接回归 mu = E[y | context]。
    训练 loss = MSE(mu, y_target)。
    推断时把同一个 mu 复制 S 份——区间宽度为 0，
    CRPS 退化为 MAE，PICP/PINAW 意义不大但数值仍然合法。
    """

    def __init__(self, env_dim: int, stoch_dim: int, hidden_dim: int,
                 T_out: int, feat_dim: int, dropout: float = 0.1):
        super().__init__(env_dim, stoch_dim, hidden_dim, T_out, feat_dim, dropout)
        self.net = self._build_net(self.cfm_dim, dropout)

    def forward(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor) -> torch.Tensor:
        """返回 mu: [B, N, cfm_dim]"""
        return self.net(self._concat(He_prime, Hs_prime))

    def loss(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
             y_target: torch.Tensor, **_) -> torch.Tensor:
        """MSE loss，**_ 吸收 cfm_loss 的其余关键字参数保持签名兼容。"""
        mu = self.forward(He_prime, Hs_prime)
        return F.mse_loss(mu, y_target)

    @torch.no_grad()
    def sample(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
               n_samples: int = 50, **_) -> torch.Tensor:
        """
        返回 [S, B, N, T_out, feat_dim]，与 DCFM.sample() 格式一致。
        确定性头没有随机性，S 份完全相同，CRPS = MAE。
        """
        training = self.training
        self.eval()
        mu = self.forward(He_prime, Hs_prime)               # [B, N, cfm_dim]
        if training:
            self.train()
        B, N, _ = mu.shape
        mu_4d = mu.reshape(B, N, self.T_out, self.feat_dim) # [B, N, T_out, feat]
        return mu_4d.unsqueeze(0).expand(n_samples, -1, -1, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# 变体 B：参数化高斯输出头
# ---------------------------------------------------------------------------

class GaussianHead(_BaseAlternativeHead):
    """
    预测均值 mu 和 log 方差 logvar，用 Gaussian NLL loss 训练。
    推断时从 N(mu, sigma²) 独立采 S 个样本，形成预测分布。

    logvar 截断在 [-6, 4]（sigma 范围约 [0.05, 7.4] 倍单位），
    避免方差爆炸或 NLL 退化成负无穷。
    """

    LOGVAR_MIN = -6.0
    LOGVAR_MAX =  4.0

    def __init__(self, env_dim: int, stoch_dim: int, hidden_dim: int,
                 T_out: int, feat_dim: int, dropout: float = 0.1):
        super().__init__(env_dim, stoch_dim, hidden_dim, T_out, feat_dim, dropout)
        # 共享 trunk，分别预测 mu 和 logvar
        in_dim = env_dim + stoch_dim
        trunk_out = hidden_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trunk_out),
            nn.SiLU(),
        )
        self.mu_head     = nn.Linear(trunk_out, self.cfm_dim)
        self.logvar_head = nn.Linear(trunk_out, self.cfm_dim)
        # 初始化：logvar 头偏置为 0，让初始 sigma ≈ 1，训练早期稳定
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.zeros_(self.logvar_head.weight)

    def forward(self, He_prime: torch.Tensor,
                Hs_prime: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (mu, logvar): 各 [B, N, cfm_dim]"""
        h      = self.trunk(self._concat(He_prime, Hs_prime))
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        return mu, logvar

    def loss(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
             y_target: torch.Tensor, **_) -> torch.Tensor:
        """
        Gaussian NLL loss（元素平均）。
        loss = 0.5 * mean( logvar + (y - mu)^2 / exp(logvar) ) + const
        """
        mu, logvar = self.forward(He_prime, Hs_prime)
        # 数值稳定版本：用 var = exp(logvar) 直接展开
        var = logvar.exp()
        nll = 0.5 * (logvar + (y_target - mu).pow(2) / var)
        return nll.mean()

    @torch.no_grad()
    def sample(self, He_prime: torch.Tensor, Hs_prime: torch.Tensor,
               n_samples: int = 50, **_) -> torch.Tensor:
        """
        从 N(mu, sigma²) 独立采 S 个样本。
        返回 [S, B, N, T_out, feat_dim]，与 DCFM.sample() 格式一致。
        """
        training = self.training
        self.eval()
        mu, logvar = self.forward(He_prime, Hs_prime)  # [B, N, cfm_dim]
        if training:
            self.train()
        sigma = (0.5 * logvar).exp()                   # [B, N, cfm_dim]
        B, N, _ = mu.shape

        # 一次性采 S 个独立噪声，避免 for 循环
        eps     = torch.randn(n_samples, B, N, self.cfm_dim, device=mu.device)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # [S, B, N, cfm_dim]

        return samples.reshape(n_samples, B, N, self.T_out, self.feat_dim).contiguous()
