"""
baselines/run_baselines.py
统一 baseline 运行入口（多步预测版），与 GridCFN 的 main.py 接口一致。

用法:
  uv run baselines/run_baselines.py --preset solar
  uv run baselines/run_baselines.py --preset electricity --models dcrnn mtgnn stid
  uv run baselines/run_baselines.py --preset weather --models ha stid --gpu_id 0
  uv run baselines/run_baselines.py --preset sdwpf --models dcrnn tsflow k2vae

所有结果保存在 result/baselines/<dataset>/<timestamp>/

修复：
  [1] load_weather 调用补全 feature_idx 参数
  [2] null_val 按数据集语义分别传入：
      Solar / SDWPF  → 传归一化零值（夜间/停机功率为 0 应 mask）
      Electricity / Weather → 传 None（无真实物理零值约束，不应 mask）
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from dataset import load_solar_energy, load_electricity, load_weather, load_sdwpf


# ── 工具函数 ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    if gpu_id < 0:
        return torch.device("cuda:0")
    if gpu_id >= n_gpu:
        raise ValueError(f"gpu_id={gpu_id} 超出范围，共 {n_gpu} 块 GPU")
    return torch.device(f"cuda:{gpu_id}")


def setup_logger(log_path: str, name: str = "baselines") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_data(cfg):
    """
    修复 [1]：load_weather 补全 feature_idx 参数。
    """
    d = cfg.data
    if d.dataset == "solar":
        return load_solar_energy(d.data_path, d.T_in, d.T_out,
                                 d.adj_threshold, d.batch_size)
    elif d.dataset == "electricity":
        return load_electricity(d.data_path, d.T_in, d.T_out,
                                d.adj_threshold, d.batch_size)
    elif d.dataset == "weather":
        # 修复：补全 feature_idx 参数，与 main.py 保持一致
        return load_weather(d.data_path, d.T_in, d.T_out,
                            d.adj_threshold, d.batch_size,
                            feature_idx=getattr(d, "weather_feature_idx", 0))
    elif d.dataset == "sdwpf":
        return load_sdwpf(d.data_path, d.T_in, d.T_out,
                          d.adj_threshold, d.batch_size)
    else:
        raise ValueError(d.dataset)


def _get_null_val(dataset: str, scaler) -> float | None:
    """
    修复 [2]：按数据集语义决定是否 mask 零值。

    Solar / SDWPF：目标变量为功率，夜间或停机时物理值为 0，
                   与"缺失值"含义不同但通常不计入误差统计（论文惯例）。
                   传归一化后的零值作为 null_val。

    Electricity / Weather：用电量 / 气象变量无真实零值约束，
                            零值是合法观测值，不应 mask。传 None。
    """
    if dataset in ("solar", "sdwpf"):
        return float(scaler.transform(np.array([0.0], dtype=np.float32))[0])
    else:
        return None


# ── 各 baseline 工厂函数 ──────────────────────────────────────────────────

def run_ha(loaders, adj, cfg, device, save_dir, logger,
           in_dim=None, num_nodes=None, scaler=None, null_val=None):
    from baselines.ha import run_ha as _ha
    logger.info("=" * 52 + "\n[HA] Historical Average")
    _, _, test_loader = loaders
    return _ha(test_loader, device, logger, scaler=scaler, null_val=null_val)


def run_var(loaders, adj, cfg, device, save_dir, logger,
            in_dim=None, num_nodes=None, scaler=None, null_val=None):
    from baselines.var_model import run_var as _var
    logger.info("=" * 52 + "\n[VAR] Vector AutoRegression (AR per node)")
    train_loader, _, test_loader = loaders
    return _var(train_loader, test_loader, cfg.data.T_in, cfg.data.T_out,
                device, logger=logger, scaler=scaler, null_val=null_val)


def run_dcrnn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.dcrnn import DCRNN
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[DCRNN] Diffusion Convolutional RNN")
    train_loader, val_loader, test_loader = loaders

    model = DCRNN(in_dim=in_dim, hidden_dim=64, n_layers=2,
                  K=2, out_dim=1, T_out=cfg.data.T_out).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    supports = DCRNN.build_supports(adj)
    supports = [s.to(device) for s in supports]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "dcrnn_best.pt"),
        logger     = logger,
        extra_forward_kwargs = {"supports": supports},
    )


def run_stgcn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.stgcn import STGCN, ChebConv
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[STGCN] Spatio-Temporal Graph Conv Net")
    train_loader, val_loader, test_loader = loaders

    model = STGCN(in_dim=in_dim, hidden_dim=64, kernel_size=3,
                  K=3, n_blocks=2, out_dim=1, T_out=cfg.data.T_out).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    L_tilde = ChebConv.compute_laplacian(adj).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "stgcn_best.pt"),
        logger     = logger,
        extra_forward_kwargs = {"L_tilde": L_tilde},
    )


def run_mtgnn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.mtgnn import MTGNN
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[MTGNN] Multivariate Time Series GNN")
    train_loader, val_loader, test_loader = loaders

    model = MTGNN(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        hidden_dim = 32,
        skip_dim   = 64,
        end_dim    = 128,
        n_layers   = 3,
        depth      = 2,
        dropout    = 0.3,
        propalpha  = 0.05,
        tanhalpha  = 3.0,
        embed_dim  = 40,
        top_k      = min(20, num_nodes - 1),
        out_dim    = 1,
        T_out      = cfg.data.T_out,
        seq_length = cfg.data.T_in,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "mtgnn_best.pt"),
        logger     = logger,
        extra_forward_kwargs = {},
    )


def run_agcrn(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.agcrn import AGCRN
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[AGCRN] Adaptive Graph Conv RNN")
    train_loader, val_loader, test_loader = loaders

    model = AGCRN(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        hidden_dim = 64,
        n_layers   = 2,
        embed_dim  = 10,
        cheb_k     = 2,
        out_dim    = 1,
        T_out      = cfg.data.T_out,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "agcrn_best.pt"),
        logger     = logger,
    )


def run_stid(loaders, adj, cfg, device, save_dir, logger,
             in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.stid import STID
    from baselines.utils import train_model

    logger.info("=" * 52 + "\n[STID] Spatial-Temporal Identity MLP")
    train_loader, val_loader, test_loader = loaders

    model = STID(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        T_in       = cfg.data.T_in,
        hidden_dim = 32,
        n_layers   = 3,
        embed_dim  = 32,
        out_dim    = 1,
        T_out      = cfg.data.T_out,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    return train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        scaler     = scaler,
        null_val   = null_val,
        max_epochs = cfg.train.max_epochs,
        patience   = cfg.train.patience,
        grad_clip  = cfg.train.grad_clip,
        save_path  = os.path.join(save_dir, "stid_best.pt"),
        logger     = logger,
    )


def run_csdi(loaders, adj, cfg, device, save_dir, logger,
             in_dim=1, num_nodes=1, scaler=None, null_val=None):
    """CSDI 训练需要同时用 x 和 y，使用自定义训练循环。"""
    from baselines.csdi import CSDI
    from baselines.utils import compute_prob_metrics, masked_mae

    logger.info("=" * 52 + "\n[CSDI] Conditional Score-based Diffusion")
    train_loader, val_loader, test_loader = loaders

    model = CSDI(
        num_nodes       = num_nodes,
        in_dim          = in_dim,
        T_in            = cfg.data.T_in,
        T_out           = cfg.data.T_out,
        channels        = 64,
        n_layers        = 4,
        nheads          = 8,
        diffusion_steps = 100,
        n_samples       = 10,
        out_dim         = 1,
    ).to(device)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    best_val, no_improve = float("inf"), 0
    patience   = cfg.train.patience
    max_epochs = cfg.train.max_epochs
    save_path  = os.path.join(save_dir, "csdi_best.pt")
    history    = {"train_loss": [], "val_mae": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        tl = float(np.mean(losses))
        history["train_loss"].append(tl)

        old_n = model.n_samples
        model.n_samples = 5
        model.eval()
        vm = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)                    # [B, T_out, N, out_dim]
                vm.append(masked_mae(pred, y, null_val).item())
        val_mae = float(np.mean(vm))
        model.n_samples = old_n
        history["val_mae"].append(val_mae)

        try:
            scheduler.step(val_mae)
        except TypeError:
            scheduler.step()

        if val_mae < best_val:
            best_val, no_improve = val_mae, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d} | train={tl:.4f} | val={val_mae:.4f} | "
                        f"best={best_val:.4f} | {time.time() - t0:.1f}s")
        if no_improve >= patience:
            logger.info(f"  Early stop @ epoch {epoch}")
            break

    # 测试
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    samples_list, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            samples_list.append(model.sample(x, return_samples=True).cpu())
            trues.append(y.cpu().permute(0, 2, 1, 3))  # [B, N, T_out, F]

    samples_all = torch.cat(samples_list, dim=1).numpy()  # [S, total, N, T_out, F]
    true_all = torch.cat(trues, dim=0).numpy()            # [total, N, T_out, F]
    null_val_eval = null_val
    if scaler is not None:
        shape = samples_all.shape
        samples_all = scaler.inverse_transform(samples_all.reshape(-1)).reshape(shape)
        true_all = scaler.inverse_transform(true_all.reshape(-1)).reshape(true_all.shape)
        null_val_eval = 0.0 if null_val is not None else None

    test_m = compute_prob_metrics(samples_all, true_all, null_val_eval)
    mae, rmse, mape = test_m["MAE"], test_m["RMSE"], test_m["MAPE"]
    logger.info(f"  [Test] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  "
                f"CRPS={test_m['CRPS']:.4f}")
    history.update({"test_metrics": test_m, "test_mae": mae,
                    "test_rmse": rmse, "test_mape": mape,
                    "test_crps": test_m["CRPS"]})
    return history


def run_tsflow(loaders, adj, cfg, device, save_dir, logger,
               in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.tsflow import run_tsflow as _tsflow
    logger.info("=" * 52 + "\n[TSFlow] CFM + GP(OU) prior (no graph)")
    return _tsflow(loaders, adj, cfg, device, save_dir, logger,
                   in_dim=in_dim, num_nodes=num_nodes,
                   scaler=scaler, null_val=null_val)


def run_k2vae(loaders, adj, cfg, device, save_dir, logger,
              in_dim=1, num_nodes=1, scaler=None, null_val=None):
    from baselines.k2vae import run_k2vae as _k2vae
    logger.info("=" * 52 + "\n[K2VAE] Koopman-Kalman VAE (no graph)")
    return _k2vae(loaders, adj, cfg, device, save_dir, logger,
                  in_dim=in_dim, num_nodes=num_nodes,
                  scaler=scaler, null_val=null_val)


# ── 模型注册表 ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "ha":     run_ha,
    "var":    run_var,
    "dcrnn":  run_dcrnn,
    "stgcn":  run_stgcn,
    "mtgnn":  run_mtgnn,
    "agcrn":  run_agcrn,
    "stid":   run_stid,
    "csdi":   run_csdi,
    "tsflow": run_tsflow,
    "k2vae":  run_k2vae,
}


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridCFN Baselines Runner (Multi-Step)")
    parser.add_argument("--preset", type=str, default="solar",
                        choices=["solar", "electricity", "weather", "sdwpf"])
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="要运行的 baseline 列表")
    parser.add_argument("--gpu_id", type=int, default=-1)
    args = parser.parse_args()

    cfg     = get_config(args.preset)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = cfg.data.dataset
    save_dir = os.path.join("result", "baselines", dataset, ts)
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(
        os.path.join(save_dir, f"baselines_{dataset}_{ts}.log"))
    set_seed(cfg.train.seed)
    device = get_device(args.gpu_id)

    logger.info(f"Dataset  : {dataset}")
    logger.info(f"T_out    : {cfg.data.T_out} (multi-step)")
    logger.info(f"Device   : {device}")
    logger.info(f"Models   : {args.models}")
    logger.info(f"Save dir : {save_dir}")

    # 加载数据
    train_loader, val_loader, test_loader, adj, scaler, in_dim = load_data(cfg)
    num_nodes = adj.shape[0]
    loaders   = (train_loader, val_loader, test_loader)
    logger.info(f"Nodes={num_nodes}, in_dim={in_dim}, T_in={cfg.data.T_in}")

    # 修复 [2]：按数据集语义决定 null_val
    null_val = _get_null_val(dataset, scaler)
    if null_val is not None:
        logger.info(f"null_val (normalized zero): {null_val:.4f}  "
                    f"(scaler mean={scaler.mean:.4f}, std={scaler.std:.4f})")
    else:
        logger.info(f"null_val: None (dataset={dataset}, no zero-masking)")

    all_results = {}

    for model_name in args.models:
        try:
            fn = MODEL_REGISTRY[model_name]
            result = fn(loaders, adj, cfg, device, save_dir, logger,
                        in_dim=in_dim, num_nodes=num_nodes,
                        scaler=scaler, null_val=null_val)
            all_results[model_name] = result
        except Exception as e:
            logger.error(f"[{model_name}] FAILED: {e}", exc_info=True)
            all_results[model_name] = {"error": str(e)}

    # ── 汇总表 ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10}")
    logger.info("-" * 60)
    for name, res in all_results.items():
        if "error" in res:
            logger.info(f"{name:<12}  ERROR: {res['error']}")
        else:
            metrics = res.get("test_metrics", {}) if isinstance(res, dict) else {}
            mae  = res.get("test_mae",  metrics.get("MAE",  float("nan")))
            rmse = res.get("test_rmse", metrics.get("RMSE", float("nan")))
            mape = res.get("test_mape", metrics.get("MAPE", float("nan")))
            logger.info(f"{name:<12} {mae:>10.4f} {rmse:>10.4f} {mape:>10.2f}")
    logger.info("=" * 60)

    # 保存 JSON
    result_path = os.path.join(save_dir,
                               f"baselines_results_{dataset}_{ts}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o)
                  if isinstance(o, (np.floating,)) else str(o))
    logger.info(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()