# DCFM

> **Disentangled Dual-Conditional Flow Matching for Spatiotemporal Energy Probabilistic Forecasting**

DCFM is a PyTorch implementation for multi-step probabilistic forecasting on spatiotemporal energy data. It disentangles historical signals into long-term evolution and short-term fluctuation components, constructs complementary conditional representations, and learns the conditional distribution of future states through conditional flow matching.

## Overview

The model contains three main components:

1. **Heterogeneous Disentangled Representation Learning (HDRL)**: adaptively decomposes the input sequence and encodes long-term and short-term components with heterogeneous graph-temporal encoders.
2. **Conditional Representation Enhancement (CRE)**: builds long-term condition `C_L` using multi-scale temporal context and short-term condition `C_S` using evolution-guided spatial message passing.
3. **Conditional Flow Matching (CFM)**: uses asymmetric dual-condition modulation to learn a conditional vector field, then generates forecast samples with a second-order ODE solver.

The training and evaluation pipeline reports both point and probabilistic metrics: MAE, RMSE, CRPS, PICP, and PINAW.

## Repository layout

```text
.
├── main.py                         # DCFM training entry point
├── model.py                        # DCFM model definition
├── train.py                        # training, sampling, calibration, and evaluation
├── config.py                       # dataset presets and hyperparameters
├── dataset.py                      # loaders and graph construction
├── analyze.py                      # calibration and case-study visualizations
├── plot_prediction.py              # comparison plot from saved predictions
├── plot_paper_result_figures.py    # paper result figures
├── wind_mask_utils.py              # optional SDWPF directional graph prior
├── ablation/                       # component and prediction-head ablations
├── baselines/                      # baseline implementations and runner
├── data/                           # datasets (not included)
└── result/                         # checkpoints, predictions, logs, and figures
```

## Environment

Python 3.9--3.11 and PyTorch are recommended. Install a PyTorch build compatible with your CUDA version first, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the PyTorch build appropriate for your platform from pytorch.org.
pip install torch
pip install numpy scipy pandas matplotlib scikit-learn tqdm tabulate
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python main.py --help
```

## Data preparation

Place the source files under `data/`. The default locations are defined in [`config.py`](config.py).

| Preset | Default file | Nodes | Sampling interval |
|---|---|---:|---:|
| `solar` | `data/solar_AL.txt` | 137 | 10 min |
| `electricity` | `data/electricity.txt` | 321 | 1 h |
| `sdwpf` | `data/sdwpf_245days_v1.csv` | 134 | 10 min |
| `pjm` | `data/archive/` | 12 | 1 h |
| `weather` | `data/weather2k.npy` | dataset-dependent | 1 h |

For SDWPF, placing `sdwpf_turb_location.csv` next to the main CSV enables the optional directional wind-mask prior. The model remains runnable when this coordinate file is unavailable.

## Training DCFM

Run from the repository root:

```bash
# Default 12-step forecast
python main.py --preset solar
python main.py --preset electricity
python main.py --preset sdwpf
python main.py --preset pjm

# Override the forecast horizon
python main.py --preset solar --T_out 24
```

Each run writes artifacts to `result/<dataset>/<timestamp>/`:

```text
dcfm_<dataset>_<timestamp>.pt  # best checkpoint
DCFM_prediction.npy            # predictive mean, [samples, T_out, nodes, features]
ground_truth.npy               # target values, [samples, T_out, nodes, features]
history_<dataset>_<timestamp>.json
dcfm_<dataset>_<timestamp>.log
```

`DCFM_prediction.npy` is the canonical output name. The plotting utility also recognizes legacy `GridCFN_prediction.npy` files so previously generated results remain usable.

## Configuration

All dataset-specific settings are centralised in [`config.py`](config.py). The main fields are:

- `DataConfig`: input history `T_in`, horizon `T_out`, adjacency threshold, and batch size.
- `ModelConfig`: graph/temporal hidden dimensions, low-rank dimension, multi-scale dilations, and CFM architecture.
- `TrainConfig`: optimizer settings, early stopping, CFM sample count, ODE steps, and temperature calibration parameters.

To add a dataset, create a loader in [`dataset.py`](dataset.py), then add a corresponding preset in `get_config()`.

## Baselines and ablations

Run a baseline with the unified runner:

```bash
python baselines/run_baselines.py --help
python baselines/run_baselines.py --preset electricity --models mtgnn
```

Run a quick ablation smoke test:

```bash
python ablation/run_ablation.py --preset solar --only full noclub --quick
```

The available ablations include heterogeneous encoding, the CLUB constraint, long-term context aggregation, evolution-guided short-term enhancement, low-rank regularization, and deterministic/Gaussian prediction heads. Results are saved under `ablation_results/<dataset>/<timestamp>/`.

## Visualisation

Create a prediction comparison figure from saved arrays:

```bash
python plot_prediction.py --help
python plot_prediction.py result/plot/electricity \
  --models DCFM MTGNN TSFlow --horizon 11 --length 48
```

Generate paper-level aggregate figures after filling the result values in the script:

```bash
python plot_paper_result_figures.py
```

For calibration or fan-chart analysis from a checkpoint:

```bash
python analyze.py --checkpoint result/solar/<timestamp>/dcfm_solar_<timestamp>.pt \
  --preset solar --output_dir result/analysis/solar
```

## Reproducibility notes

- Training uses the random seed in `TrainConfig` (default: `42`).
- All reported point and probabilistic metrics are computed after inverse normalization.
- The validation set is used to select the checkpoint and calibrate the predictive sample spread; the test set is evaluated only after training is complete.
- Data files, checkpoints, and generated result folders are intentionally not versioned with the source code.

## Citation

If you use this implementation, please cite the accompanying DCFM paper after bibliographic information is finalized.
