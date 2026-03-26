"""
GridCFN 单元测试
运行: python test_model.py

修复说明：
  - test_disentangler：验证新的返回值 (He, Hs, He_seq)，He_seq=[B,T,N,De]
  - test_scgmp：env_dim 改为 ms_out_dim（Bug-5 修复，传 He_prime）
  - test_full_model：compute_loss 新增 lambda_mi 参数（Bug-4）
  - test_backward：拆分主网络 / MINE 两步 backward（Bug-1）
  - test_config：验证 warmup_epochs 字段存在（Bug-4）
"""

import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from model import (GCN, TCN, Backbone, CausalDisentangler, MINEEstimator,
                   MultiScaleContext, SCGMP, ProbabilisticPredictor,
                   GridCFN, nll_gaussian_loss)
from dataset import make_synthetic_dataset, Scaler
from train import evaluate_all


def make_adj(N=10, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    adj = (rng.random((N, N)) > (1 - density)).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    return torch.tensor(adj)


def test_gcn():
    print("Testing GCN...")
    B, N, F, D = 4, 10, 8, 16
    x   = torch.randn(B, N, F)
    adj = GridCFN.normalize_adj(make_adj(N))
    out = GCN(F, 32, D, n_layers=2)(x, adj)
    assert out.shape == (B, N, D)
    print(f"  OK: {out.shape}")


def test_tcn():
    print("Testing TCN...")
    B, N, T, F, D = 4, 10, 12, 8, 16
    x   = torch.randn(B, N, T, F)
    out = TCN(F, D, n_layers=4)(x)
    assert out.shape == (B, N, T, D)
    print(f"  OK: {out.shape}")


def test_backbone():
    print("Testing Backbone...")
    B, T, N, F = 2, 12, 10, 5
    x   = torch.randn(B, T, N, F)
    adj = GridCFN.normalize_adj(make_adj(N))
    H   = Backbone(F, 32, 64)(x, adj)
    assert H.shape == (B, T, N, 64)
    print(f"  OK: {H.shape}")


def test_disentangler():
    print("Testing CausalDisentangler...")
    B, T, N, D = 2, 12, 10, 64
    H = torch.randn(B, T, N, D)
    # [Bug-2 修复] 现在返回 (He, Hs, He_seq)
    He, Hs, He_seq = CausalDisentangler(D, 32, 32)(H)
    assert He.shape == (B, N, 32), f"He shape: {He.shape}"
    assert Hs.shape == (B, N, 32), f"Hs shape: {Hs.shape}"
    assert He_seq.shape == (B, T, N, 32), f"He_seq shape: {He_seq.shape}"
    print(f"  OK: He={He.shape}, Hs={Hs.shape}, He_seq={He_seq.shape}")


def test_mine():
    print("Testing MINEEstimator...")
    B, N = 4, 10
    He = torch.randn(B, N, 32)
    Hs = torch.randn(B, N, 32)
    mi = MINEEstimator(32, 32)(He, Hs)
    assert mi.ndim == 0
    print(f"  OK: MI={mi.item():.4f}")


def test_ms_context():
    print("Testing MultiScaleContext...")
    B, T, N, De = 2, 12, 10, 32
    out = MultiScaleContext(De, 32)(torch.randn(B, T, N, De))
    assert out.shape == (B, N, 32)
    print(f"  OK: {out.shape}")


def test_scgmp():
    print("Testing SCGMP...")
    B, N, Ds = 2, 10, 32
    ms_out_dim = 32  # [Bug-5 修复] env_dim = ms_out_dim，传 He_prime
    Hs       = torch.randn(B, N, Ds)
    He_prime = torch.randn(B, N, ms_out_dim)
    edge_index = GridCFN.adj_to_edge_index(make_adj(N))
    out = SCGMP(Ds, ms_out_dim, n_layers=3)(Hs, He_prime, edge_index)
    assert out.shape == (B, N, Ds)
    print(f"  OK: {out.shape}")


def test_predictor():
    print("Testing ProbabilisticPredictor...")
    B, N = 2, 10
    mu, sigma = ProbabilisticPredictor(64, 1)(torch.randn(B, N, 64))
    assert mu.shape == (B, N, 1) and (sigma > 0).all()
    print(f"  OK: mu={mu.shape}, sigma>0: True")


def test_full_model():
    print("Testing full GridCFN forward pass...")
    cfg = get_config("debug")
    m   = cfg.model
    B, T, N, F = 2, cfg.data.T_in, cfg.data.synthetic_N, m.in_dim
    x   = torch.randn(B, T, N, F)
    adj = make_adj(N, density=0.3)
    model = GridCFN(in_dim=m.in_dim, gcn_hidden=m.gcn_hidden,
                    gcn_layers=m.gcn_layers, tcn_hidden=m.tcn_hidden,
                    tcn_layers=m.tcn_layers, env_dim=m.env_dim,
                    stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
                    n_scg_layers=m.n_scg_layers, out_dim=m.out_dim,
                    lambda_mi=m.lambda_mi)
    mu, sigma, mi_loss = model(x, adj)
    assert mu.shape == (B, N, 1)
    y = torch.randn(B, N, 1)
    # [Bug-4 修复] 传入动态 lambda_mi
    loss, l_nll, l_mi = model.compute_loss(mu, sigma, y, mi_loss, lambda_mi=0.5)
    assert not torch.isnan(loss)
    print(f"  OK: loss={loss.item():.4f}, nll={l_nll.item():.4f}, mi={l_mi.item():.4f}")


def test_backward():
    print("Testing backward pass (MINE minimax)...")
    cfg = get_config("debug")
    m   = cfg.model
    B, T, N = 2, cfg.data.T_in, cfg.data.synthetic_N
    x   = torch.randn(B, T, N, m.in_dim)
    adj = make_adj(N)
    y   = torch.randn(B, N, 1)
    model = GridCFN(in_dim=m.in_dim, gcn_hidden=m.gcn_hidden,
                    gcn_layers=m.gcn_layers, tcn_hidden=m.tcn_hidden,
                    tcn_layers=m.tcn_layers, env_dim=m.env_dim,
                    stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
                    n_scg_layers=m.n_scg_layers, out_dim=m.out_dim,
                    lambda_mi=m.lambda_mi)

    # [Bug-1 修复] MINE 独立 optimizer，主网络独立 optimizer
    from train import _forward_for_mine
    mine_opt = torch.optim.Adam(model.mine_parameters(), lr=2e-3)
    main_opt = torch.optim.Adam(model.main_parameters(), lr=1e-3)

    # Step1: MINE 梯度上升
    mine_opt.zero_grad()
    mi_est = _forward_for_mine(model, x, adj)
    (-mi_est).backward()
    mine_opt.step()

    # Step2: 主网络梯度下降
    main_opt.zero_grad()
    mu, sigma, mi_loss = model(x, adj)
    loss, _, _ = model.compute_loss(mu, sigma, y, mi_loss, lambda_mi=0.5)
    loss.backward()
    main_opt.step()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
    print(f"  OK: loss={loss.item():.4f}")


def test_metrics():
    print("Testing metrics...")
    N  = 100
    mu = np.random.randn(N, 10, 1)
    sigma = np.abs(np.random.randn(N, 10, 1)) + 0.1
    y  = mu + sigma * np.random.randn(N, 10, 1) * 0.5
    metrics = evaluate_all(mu, sigma, y)
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} is not finite"
    print(f"  OK: {metrics}")


def test_dataset():
    print("Testing synthetic dataset...")
    cfg = get_config("debug")
    d   = cfg.data
    train_loader, val_loader, test_loader, adj, scaler = make_synthetic_dataset(
        T=d.synthetic_T, N=d.synthetic_N, F=d.synthetic_F,
        T_in=d.T_in, batch_size=d.batch_size, seed=cfg.train.seed,
    )
    x, y = next(iter(train_loader))
    assert x.shape[1] == d.T_in and x.shape[-1] == d.synthetic_F
    print(f"  OK: x={x.shape}, y={y.shape}, adj={adj.shape}")


def test_config():
    print("Testing Config...")
    for preset in ["default", "solar", "electricity", "weather", "debug"]:
        cfg = get_config(preset)
        assert cfg.data.dataset is not None
        assert cfg.model.lambda_mi > 0
        assert hasattr(cfg.train, 'warmup_epochs'), "warmup_epochs missing"
    print("  OK: all presets load correctly")


def run_mini_training():
    """端到端 mini 训练（debug preset）。"""
    print("\nRunning mini end-to-end training (debug preset)...")
    from train import train

    cfg    = get_config("debug")
    device = torch.device("cpu")

    train_loader, val_loader, test_loader, adj, scaler = make_synthetic_dataset(
        T=cfg.data.synthetic_T, N=cfg.data.synthetic_N, F=cfg.data.synthetic_F,
        T_in=cfg.data.T_in, batch_size=cfg.data.batch_size, seed=cfg.train.seed,
    )
    m = cfg.model
    model = GridCFN(in_dim=m.in_dim, gcn_hidden=m.gcn_hidden,
                    gcn_layers=m.gcn_layers, tcn_hidden=m.tcn_hidden,
                    tcn_layers=m.tcn_layers, env_dim=m.env_dim,
                    stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
                    n_scg_layers=m.n_scg_layers, out_dim=m.out_dim,
                    lambda_mi=m.lambda_mi).to(device)

    history = train(model, train_loader, val_loader, test_loader,
                    adj, device, cfg_train=cfg.train, scaler=scaler)
    print("  OK: mini training completed.")
    return history


if __name__ == "__main__":
    tests = [
        test_config, test_gcn, test_tcn, test_backbone,
        test_disentangler, test_mine, test_ms_context,
        test_scgmp, test_predictor, test_full_model,
        test_backward, test_metrics, test_dataset,
    ]

    print("=" * 52)
    print("GridCFN Unit Tests")
    print("=" * 52)

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*52}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        run_mini_training()
