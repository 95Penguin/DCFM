"""
baselines/diffstg.py
DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with
Denoising Diffusion Models — Wen et al., SIGSPATIAL 2023
"""
import math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class DDPMScheduler:
    """quad beta schedule"""
    def __init__(self, N=200, beta_start=1e-4, beta_end=0.02, schedule='quad'):
        self.N = N
        t = torch.linspace(0, 1, N)
        if schedule == 'quad':
            beta = beta_start + (beta_end - beta_start) * t**2
        else:
            beta = torch.linspace(beta_start, beta_end, N)
        alpha_hat = 1.0 - beta
        alpha_bar = torch.cumprod(alpha_hat, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])
        self.beta, self.alpha_hat, self.alpha_bar = beta, alpha_hat, alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    def to(self, device):
        for a in ['beta','alpha_hat','alpha_bar','alpha_bar_prev','sqrt_alpha_bar','sqrt_one_minus_alpha_bar']:
            setattr(self, a, getattr(self, a).to(device))
        return self

    def q_sample(self, x0, n, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa, soa = self.sqrt_alpha_bar[n], self.sqrt_one_minus_alpha_bar[n]
        while sa.dim() < x0.dim():
            sa, soa = sa.unsqueeze(-1), soa.unsqueeze(-1)
        return sa * x0 + soa * noise


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim=64, proj_dim=128):
        super().__init__()
        if dim % 2:
            dim += 1
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.SiLU(), nn.Linear(proj_dim, proj_dim))
    def forward(self, n):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=n.device) / max(half - 1, 1))
        angles = n.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=-1))


class GCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
    def forward(self, h, agcn):
        return torch.einsum('nm,bmc->bnc', agcn, self.linear(h))


class STResidualBlock(nn.Module):
    def __init__(self, channels, diff_emb_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size - 1
        self.tcn = nn.Conv1d(channels, 2*channels, kernel_size, padding=pad)
        self.gcn = GCNLayer(channels, channels)
        self.diff_proj = nn.Linear(diff_emb_dim, 2*channels)
        self.norm = nn.LayerNorm(channels)
    def forward(self, h, agcn, diff_emb):
        B, C, N, T = h.shape
        residual = h
        h_t = h.reshape(B*N, C, T)
        h_t = self.tcn(h_t)[..., :T]
        P, Q = h_t.chunk(2, dim=1)
        h = (P * torch.sigmoid(Q)).reshape(B, C, N, T)
        sc = self.diff_proj(diff_emb)
        scale, shift = sc.chunk(2, dim=-1)
        h = h * (1.0 + scale[:,:,None,None]) + shift[:,:,None,None]
        h_s = h.permute(0,3,2,1).reshape(B*T, N, C)
        h_s = F.gelu(self.gcn(h_s, agcn))
        h = h_s.reshape(B, T, N, C).permute(0,3,2,1)
        h = self.norm(h.permute(0,2,3,1)).permute(0,3,2,1)
        return h + residual


class UGnet(nn.Module):
    def __init__(self, out_feat, hidden_size, T_all, n_nodes, diff_emb_dim=128, n_down=2, kernel_size=3):
        super().__init__()
        self.n_down = n_down
        self.input_proj = nn.Conv2d(2*out_feat, hidden_size, 1)
        self.diff_emb = DiffusionEmbedding(dim=64, proj_dim=diff_emb_dim)
        self.down_blocks = nn.ModuleList([STResidualBlock(hidden_size, diff_emb_dim, kernel_size) for _ in range(n_down)])
        self.downsample = nn.ModuleList([nn.AvgPool2d((1,2), stride=(1,2)) for _ in range(n_down)])
        self.mid_block = STResidualBlock(hidden_size, diff_emb_dim, kernel_size)
        self.upsample = nn.ModuleList([nn.Upsample(scale_factor=(1,2), mode='nearest') for _ in range(n_down)])
        self.skip_merge = nn.ModuleList([nn.Conv2d(2*hidden_size, hidden_size, 1) for _ in range(n_down)])
        self.up_blocks = nn.ModuleList([STResidualBlock(hidden_size, diff_emb_dim, kernel_size) for _ in range(n_down)])
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, hidden_size), hidden_size), nn.SiLU(), nn.Conv2d(hidden_size, out_feat, 1))
    def forward(self, x_n, x_msk, n, agcn):
        h = self.input_proj(torch.cat([x_n, x_msk], dim=1))
        diff_emb = self.diff_emb(n)
        skips = []
        for blk, ds in zip(self.down_blocks, self.downsample):
            h = blk(h, agcn, diff_emb)
            skips.append(h)
            h = ds(h)
        h = self.mid_block(h, agcn, diff_emb)
        for us, merge, blk, skip in zip(self.upsample, self.skip_merge, self.up_blocks, reversed(skips)):
            h = us(h)
            if h.shape[-1] > skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            elif h.shape[-1] < skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = merge(torch.cat([h, skip], dim=1))
            h = blk(h, agcn, diff_emb)
        return self.output_proj(h)


class DiffSTG(nn.Module):
    def __init__(self, in_dim, out_feat, T_in, T_out, n_nodes,
                 hidden_size=32, N=200, diff_emb_dim=128, n_down=2, kernel_size=3,
                 beta_start=1e-4, beta_end=0.02, schedule='quad'):
        super().__init__()
        self.in_dim, self.out_feat = in_dim, out_feat
        self.T_in, self.T_out, self.T_all = T_in, T_out, T_in + T_out
        self.N = N
        self.scheduler = DDPMScheduler(N, beta_start, beta_end, schedule)
        self.ugnet = UGnet(out_feat, hidden_size, T_in+T_out, n_nodes, diff_emb_dim, n_down, kernel_size)

    @staticmethod
    def build_agcn(adj):
        A = adj + torch.eye(adj.shape[0], device=adj.device)
        d = A.sum(dim=1).clamp(min=1e-6)
        d_inv = d.pow(-0.5)
        return d_inv.unsqueeze(1) * A * d_inv.unsqueeze(0)

    def _x_msk(self, x, y):
        """mask: 历史区保留x[..., :out_feat], 未来区置0"""
        x = x[..., :self.out_feat]  # only keep predicted features
        x_all = torch.cat([x, y], dim=1)
        mask = torch.ones_like(x_all)
        mask[:, self.T_in:] = 0.0
        return (x_all * mask).permute(0,3,2,1).contiguous()

    def compute_loss(self, x, y, agcn):
        B, device = x.shape[0], x.device
        x0 = torch.cat([x[..., :self.out_feat], y], dim=1).permute(0,3,2,1).contiguous()
        msk = self._x_msk(x, y)
        n = torch.randint(0, self.N, (B,), device=device)
        eps = torch.randn_like(x0)
        xn = self.scheduler.q_sample(x0, n, eps)
        return F.mse_loss(self.ugnet(xn, msk, n, agcn), eps)

    @torch.no_grad()
    def sample(self, x, agcn, n_samples=50, n_steps=200):
        B, N, F = x.shape[0], x.shape[2], self.out_feat
        S, device = n_samples, x.device
        sched = self.scheduler
        y_zero = torch.zeros(B, self.T_out, N, F, device=device)
        msk = self._x_msk(x, y_zero).repeat_interleave(S, dim=0)
        step_idx = torch.linspace(0, self.N-1, n_steps, dtype=torch.long, device=device)
        x_cur = torch.randn(S*B, F, N, self.T_all, device=device)
        for i in reversed(range(n_steps)):
            n_cur, n_vec = step_idx[i], step_idx[i].expand(S*B)
            eps_pred = self.ugnet(x_cur, msk, n_vec, agcn)
            ab = sched.alpha_bar[n_cur]
            b = sched.beta[n_cur]
            ah = sched.alpha_hat[n_cur]
            x0_pred = ((x_cur - (1-ab).sqrt()*eps_pred) / ab.sqrt().clamp(min=1e-8)).clamp(-3,3)
            if i > 0:
                ab_prev = sched.alpha_bar[step_idx[i-1]]
                coef1 = ab_prev.sqrt()*b / (1-ab).clamp(1e-8)
                coef2 = ah.sqrt()*(1-ab_prev) / (1-ab).clamp(1e-8)
                mu = coef1*x0_pred + coef2*x_cur
                var = (1-ab_prev)/(1-ab).clamp(1e-8)*b
                x_cur = mu + var.sqrt()*torch.randn_like(mu)
            else:
                x_cur = x0_pred
        xp = x_cur[..., self.T_in:].reshape(S,B,F,N,self.T_out).permute(0,1,3,4,2)
        return xp.contiguous()


def _evaluate_diffstg(model, loader, device, agcn, scaler, n_samples, n_steps, null_val=None):
    from baselines.utils import compute_prob_metrics
    model.eval()
    sl, yl = [], []
    for x, y in loader:
        x = x.to(device)
        raw = model.sample(x, agcn, n_samples=n_samples, n_steps=n_steps)
        sl.append(raw.cpu().numpy())
        yl.append(y.permute(0,2,1,3).numpy())
    s = np.concatenate(sl, axis=1)
    y = np.concatenate(yl, axis=0)
    m_norm = compute_prob_metrics(s, y)
    if scaler is not None:
        shape = s.shape
        s_mean = scaler.mean[..., :1] if scaler.mean.shape[-1] > 1 else scaler.mean
        s_std  = scaler.std[..., :1]  if scaler.std.shape[-1] > 1  else scaler.std
        s = (s.reshape(-1)*s_std + s_mean).reshape(shape)
        y = (y.reshape(-1)*s_std + s_mean).reshape(y.shape)
    m = compute_prob_metrics(s, y)
    for k,v in m_norm.items():
        m[f'{k}_norm'] = v
    return m


def run_diffstg(loaders, adj, cfg, device, save_dir, logger,
                in_dim=None, num_nodes=None, scaler=None, null_val=None):
    train_loader, val_loader, test_loader = loaders
    d, m, t_cfg = cfg.data, cfg.model, cfg.train
    in_dim = in_dim or 1
    num_nodes = num_nodes or adj.shape[0]
    # infer out_feat from data
    for _, y_batch in train_loader:
        out_feat = y_batch.shape[3]
        break
    hidden_size_cfg = getattr(m, 'gcn_hidden', 32)
    if num_nodes > 300:
        hidden_size = min(hidden_size_cfg, 32)
    else:
        hidden_size = min(hidden_size_cfg, 64)
    N_diff = getattr(t_cfg, 'diffstg_N', 200)
    n_steps_val = getattr(t_cfg, 'diffstg_n_steps_val', 50)
    n_steps_test = getattr(t_cfg, 'diffstg_n_steps_test', 200)
    n_samples_val = getattr(t_cfg, 'cfm_n_samples', 10)
    n_samples_test = getattr(t_cfg, 'cfm_n_samples_test', 50)
    save_path = os.path.join(save_dir, 'diffstg_best.pt')
    model = DiffSTG(
        in_dim=in_dim, out_feat=out_feat, T_in=d.T_in, T_out=d.T_out,
        n_nodes=num_nodes, hidden_size=hidden_size, N=N_diff,
        diff_emb_dim=128, n_down=2, kernel_size=3,
        beta_start=1e-4, beta_end=0.02, schedule='quad',
    ).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    agcn = DiffSTG.build_agcn(adj_t).to(device)
    model.scheduler.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[DiffSTG] Parameters: {n_params:,}')
    logger.info(f'[DiffSTG] in_dim={in_dim}, out_feat={out_feat}, hidden={hidden_size}, N={N_diff}')
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=t_cfg.lr_decay_factor, patience=t_cfg.lr_decay_patience)
    best_val_crps = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_crps': [], 'val_mae': [], 'val_rmse': []}
    logger.info('[DiffSTG] 开始训练 (图感知 DDPM + UGnet)')
    logger.info(f"{'Epoch':>6} | {'Loss':>8} | {'Val MAE':>8} | {'Val RMSE':>9} | {'Val CRPS':>9} | {'LR':>8} | {'Time':>6}")
    for epoch in range(1, t_cfg.max_epochs+1):
        t0 = time.time()
        model.train()
        total_loss, n_batches, nan_skipped, n_total = 0.0, 0, 0, len(train_loader)
        log_every = max(1, n_total // 5)
        for bi, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            loss = model.compute_loss(x, y, agcn)
            if not torch.isfinite(loss):
                nan_skipped += 1
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if bi % log_every == 0:
                logger.info(f'  [DiffSTG] Epoch {epoch} [{bi}/{n_total}] loss={total_loss/max(n_batches,1):.4f}')
        avg_loss = total_loss / max(n_batches, 1)
        val_m = _evaluate_diffstg(model, val_loader, device, agcn, scaler, n_samples_val, n_steps_val, null_val)
        scheduler_lr.step(val_m["CRPS"])
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history['train_loss'].append(avg_loss)
        history['val_crps'].append(val_m["CRPS"])
        history['val_mae'].append(val_m["MAE"])
        history['val_rmse'].append(val_m["RMSE"])
        nan_warn = f'  [!] {nan_skipped} NaN skipped' if nan_skipped else ''
        logger.info(f'{epoch:>6} | {avg_loss:>8.4f} | {val_m["MAE"]:>8.4f} | {val_m["RMSE"]:>9.4f} | {val_m["CRPS"]:>9.4f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s{nan_warn}')
        if val_m["CRPS"] < best_val_crps:
            best_val_crps = val_m["CRPS"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= t_cfg.patience:
                logger.info(f'[DiffSTG] 早停于 epoch {epoch}')
                break
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_m = _evaluate_diffstg(model, test_loader, device, agcn, scaler, n_samples_test, n_steps_test, null_val)
    sep = "="*55
    logger.info(f"\n{sep}")
    logger.info(f'[DiffSTG] TEST SET RESULTS (归一化域, T_out={d.T_out})')
    logger.info(sep)
    for k in ["MAE","RMSE","CRPS","PICP","PINAW"]:
        logger.info(f'  {k:<8}: {test_m[f"{k}_norm"]:.4f}')
    logger.info(sep)
    logger.info(f'[DiffSTG] TEST SET RESULTS (反归一化域, T_out={d.T_out})')
    logger.info(sep)
    for k in ["MAE","RMSE","CRPS","PICP","PINAW"]:
        logger.info(f'  {k:<8}: {test_m[k]:.4f}')
    logger.info(f'  {chr(39)+"Step"+chr(39):<6}  {chr(39)+"MAE"+chr(39):>8}  {chr(39)+"RMSE"+chr(39):>8}  {chr(39)+"CRPS"+chr(39):>8}')
    for h in range(d.T_out):
        logger.info(f'  h={h+1:<4}  {test_m.get(f"MAE_h{h+1}",float("nan")):>8.4f}  {test_m.get(f"RMSE_h{h+1}",float("nan")):>8.4f}  {test_m.get(f"CRPS_h{h+1}",float("nan")):>8.4f}')
    logger.info(sep)
    history.update({'test_metrics': test_m, 'test_mae': test_m["MAE"], 'test_rmse': test_m["RMSE"], 'test_mape': test_m["MAPE"], 'test_crps': test_m["CRPS"]})
    return history