"""
baselines/k2vae.py
K²VAE: Koopman-Kalman Enhanced Variational AutoEncoder
for Probabilistic Time Series Forecasting  — Wu et al., ICML 2025 Spotlight
github:https://github.com/decisionintelligence/K2VAE

多步预测版: T_out 支持，VAE 单步生成（无 ODE/SDE 迭代）。

修复:
  [3] PatchEmbedding.forward 中 unfold → permute → reshape 链条缺少
      .contiguous()，导致 reshape 在非连续内存上操作，某些 PyTorch
      版本下静默返回错误视图（内存布局依赖 stride=0 的 expand 轴）。
      在 permute 后、reshape 前插入 .contiguous() 保证内存连续。
"""
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """时序 Patch 化 + 线性投影 → token 序列"""
    def __init__(self, in_dim: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj = nn.Linear(patch_len * in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)  # [B, n_p, N, F, patch_len]
        n_p = patches.shape[1]
        patches = patches.permute(0, 2, 1, 4, 3)            # [B, N, n_p, patch_len, F]
        # 修复 [3]：permute 后内存非连续，reshape 前必须 contiguous()，
        # 否则在某些 PyTorch 版本下 reshape 会基于错误的内存步长生成视图。
        patches = patches.contiguous().reshape(B, N, n_p, self.patch_len * F)
        return self.proj(patches)


# ── KoopmanNet ─────────────────────────────────────────────────────────────

class KoopmanNet(nn.Module):
    """测量函数 g(·) + 全局 Koopman 算子 K，逐步线性演化 z_{t+1}=K@z_t"""
    def __init__(self, d_model: int, koopman_dim: int):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(d_model,    koopman_dim * 2), nn.GELU(),
            nn.Linear(koopman_dim * 2, koopman_dim),
        )
        self.K = nn.Parameter(torch.eye(koopman_dim) +
                               0.01 * torch.randn(koopman_dim, koopman_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.g(tokens)
        B, N, n_p, K = z.shape
        z_flat = z.reshape(B * N, n_p, K)
        evolved = [z_flat[:, 0]]
        for t in range(1, n_p):
            evolved.append(evolved[-1] @ self.K.T)
        z_evolved = torch.stack(evolved, dim=1)
        return z_evolved.reshape(B, N, n_p, K)


# ── KalmanNet ──────────────────────────────────────────────────────────────

class KalmanNet(nn.Module):
    """GRU 近似 Kalman 滤波 → q(z|x) = N(mu, diag(sigma²))"""
    def __init__(self, koopman_dim: int, latent_dim: int, gru_hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(koopman_dim, gru_hidden, batch_first=True)
        self.mu_proj      = nn.Linear(gru_hidden, latent_dim)
        self.log_var_proj = nn.Linear(gru_hidden, latent_dim)

    def forward(self, z_k: torch.Tensor) -> tuple:
        B, N, n_p, K = z_k.shape
        z_flat = z_k.reshape(B * N, n_p, K)
        h, _ = self.gru(z_flat)
        h_last = h[:, -1, :]
        mu      = self.mu_proj(h_last).reshape(B, N, -1)
        log_var = self.log_var_proj(h_last).reshape(B, N, -1)
        log_var = log_var.clamp(-10, 2)
        return mu, log_var


# ── Decoder ────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """z → y_pred [B, N, T_out * feat_dim]"""
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── K²VAE 主模型 ──────────────────────────────────────────────────────────

class K2VAE(nn.Module):
    """Koopman-Kalman 增强 VAE，无图结构，单步生成"""
    def __init__(self, in_dim: int = 1, T_in: int = 168, T_out: int = 12,
                 out_feat: int = 1,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128,
                 koopman_dim: int = 64, latent_dim: int = 64,
                 dec_hidden: int = 256, gru_hidden: int = 64,
                 beta: float = 1.0):
        super().__init__()
        self.T_out    = T_out
        self.feat_dim = out_feat
        self.cfm_dim  = T_out * out_feat
        self.beta     = beta

        self.embedder    = PatchEmbedding(in_dim, patch_len, stride, d_model)
        self.koopman_net = KoopmanNet(d_model, koopman_dim)
        self.kalman_net  = KalmanNet(koopman_dim, latent_dim, gru_hidden)
        self.decoder     = Decoder(latent_dim, dec_hidden, self.cfm_dim)

    def encode(self, x_past: torch.Tensor):
        tokens = self.embedder(x_past)
        z_k    = self.koopman_net(tokens)
        return self.kalman_net(z_k)

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def elbo_loss(self, x_past: torch.Tensor,
                  y_target: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        y_pred = self.decoder(z)
        recon = F.mse_loss(y_pred, y_target)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        return recon + self.beta * kl

    @torch.no_grad()
    def sample(self, x_past: torch.Tensor, n_samples: int = 50,
               **kwargs) -> torch.Tensor:
        B = x_past.shape[0]
        N = x_past.shape[2]
        S = n_samples
        device = x_past.device

        mu, log_var = self.encode(x_past)
        mu_rep = mu.repeat_interleave(S, dim=0)
        lv_rep = log_var.repeat_interleave(S, dim=0)

        z = self.reparameterize(mu_rep, lv_rep)
        y = self.decoder(z)

        y = y.reshape(B, S, N, self.T_out, self.feat_dim)
        return y.permute(1, 0, 2, 3, 4).contiguous()


# ── 内部评估 ────────────────────────────────────────────────────────────────

def _evaluate_k2vae(model: K2VAE, loader, device, scaler,
                    n_samples: int, null_val: float = None) -> dict:
    from baselines.utils import compute_prob_metrics

    model.eval()
    samples_list, y_list = [], []

    for x, y in loader:
        x = x.to(device)
        raw = model.sample(x, n_samples=n_samples)
        samples_list.append(raw.cpu().numpy())
        y_list.append(y.permute(0, 2, 1, 3).numpy())

    samples_all = np.concatenate(samples_list, axis=1)
    y_all       = np.concatenate(y_list,       axis=0)

    metrics_norm = compute_prob_metrics(samples_all, y_all)

    if scaler is not None:
        shape = samples_all.shape
        s_mean = scaler.mean[..., :1] if scaler.mean.shape[-1] > 1 else scaler.mean
        s_std  = scaler.std[..., :1]  if scaler.std.shape[-1] > 1  else scaler.std
        samples_all = (samples_all.reshape(-1) * s_std + s_mean).reshape(shape)
        y_all = (y_all.reshape(-1) * s_std + s_mean).reshape(y_all.shape)

    metrics = compute_prob_metrics(samples_all, y_all)
    for k, v in metrics_norm.items():
        metrics[f"{k}_norm"] = v
    return metrics


# ── 训练入口（baselines 统一签名）─────────────────────────────────────────

def run_k2vae(loaders, adj, cfg, device, save_dir, logger,
              in_dim=None, num_nodes=None, scaler=None, null_val=None):
    """
    K²VAE 训练 + 测试。
    loaders = (train_loader, val_loader, test_loader)
    """
    train_loader, val_loader, test_loader = loaders
    d, m, t_cfg = cfg.data, cfg.model, cfg.train
    # 推断输出特征维度：SlidingWindowDataset 的 y 取 data[..., :1]，F_out = 1
    for x_batch, y_batch in train_loader:
        out_feat = y_batch.shape[3]
        break

    patch_len = min(16, d.T_in // 4)
    stride    = patch_len // 2

    model = K2VAE(
        in_dim      = in_dim if in_dim is not None else 1,
        out_feat    = out_feat,
        T_in        = d.T_in,
        T_out       = d.T_out,
        patch_len   = patch_len,
        stride      = stride,
        d_model     = getattr(m, "gcn_hidden", 64) * 2,
        koopman_dim = getattr(m, "env_dim", 32) * 2,
        latent_dim  = getattr(m, "stoch_dim", 32) * 2,
        dec_hidden  = getattr(m, "cfm_hidden", 256),
        gru_hidden  = getattr(m, "tcn_hidden", 64),
        beta        = 1.0,
    ).to(device)
    logger.info(f"[K2VAE] in_dim={in_dim}, out_feat={out_feat}, cfm_dim={model.cfm_dim}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[K2VAE] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=t_cfg.lr_decay_factor,
        patience=t_cfg.lr_decay_patience)

    n_samples_val  = getattr(t_cfg, "cfm_n_samples",      50)
    n_samples_test = getattr(t_cfg, "cfm_n_samples_test", 200)
    save_path      = os.path.join(save_dir, "k2vae_best.pt")

    best_val_crps  = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_crps": [], "val_mae": [], "val_rmse": []}

    logger.info("[K2VAE] 开始训练 (无图结构, VAE 单步生成)")
    logger.info(f"{'Epoch':>6} | {'ELBO':>8} | {'Val MAE':>8} | "
                f"{'Val RMSE':>9} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")

    for epoch in range(1, t_cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            B, T_out, N, F = y.shape
            y_flat = y.permute(0, 2, 1, 3).reshape(B, N, T_out * F)

            loss = model.elbo_loss(x, y_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        val_m = _evaluate_k2vae(model, val_loader, device, scaler,
                                n_samples_val, null_val=null_val)
        scheduler.step(val_m["CRPS"])
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["val_crps"].append(val_m["CRPS"])
        history["val_mae"].append(val_m["MAE"])
        history["val_rmse"].append(val_m["RMSE"])

        logger.info(
            f"{epoch:>6} | {avg_loss:>8.4f} | {val_m['MAE']:>8.4f} | "
            f"{val_m['RMSE']:>9.4f} | {val_m['CRPS']:>9.4f} | "
            f"{cur_lr:>8.2e} | {elapsed:>5.1f}s")

        if val_m["CRPS"] < best_val_crps:
            best_val_crps     = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t_cfg.patience:
                logger.info(f"[K2VAE] 早停于 epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))
    test_m = _evaluate_k2vae(model, test_loader, device, scaler,
                             n_samples_test, null_val=null_val)

    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info(f"[K2VAE] TEST SET RESULTS (归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[f'{k}_norm']:.4f}")
    logger.info(sep)
    logger.info(f"[K2VAE] TEST SET RESULTS (反归一化域, T_out={d.T_out})")
    logger.info(sep)
    for k in ["MAE", "RMSE", "CRPS", "PICP", "PINAW"]:
        logger.info(f"  {k:<8}: {test_m[k]:.4f}")
    logger.info(f"\n  {'Step':<6}  {'MAE':>8}  {'RMSE':>8}  {'CRPS':>8}")
    for h in range(d.T_out):
        mae_h  = test_m.get(f"MAE_h{h+1}",  float("nan"))
        rmse_h = test_m.get(f"RMSE_h{h+1}", float("nan"))
        crps_h = test_m.get(f"CRPS_h{h+1}", float("nan"))
        logger.info(f"  h={h+1:<4}  {mae_h:>8.4f}  {rmse_h:>8.4f}  {crps_h:>8.4f}")
    logger.info(sep)

    history["test_metrics"] = test_m
    history["test_mae"]  = test_m["MAE"]
    history["test_rmse"] = test_m["RMSE"]
    history["test_mape"] = test_m["MAPE"]
    return history
