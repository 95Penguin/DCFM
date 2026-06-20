# ablation_model.py
"""
GridCFN 消融实验模型包装层
────────────────────────────────────────────────────────────────────────
设计原则：不修改 model.py 任何一行，而是在外层包一层 AblationGridCFN，
通过开关决定每个待消融模块是"正常工作"还是"被替换成恒等/直通操作"。

这样做的好处：
  1. 原始 GridCFN 代码保持不变，不会因为消融实验引入 bug 污染主线代码；
  2. 每个开关关闭后，模型的输入输出张量形状完全不变，可以直接复用
     train.py 里现成的 train_one_epoch / evaluate 函数，无需改动训练循环；
  3. 关闭某个模块时，该模块的参数仍然会被创建（forward 不调用而已），
     这是为了让消融实验之间的代码路径完全一致、只有"是否使用"这一个变量，
     避免因为参数量变化引入额外的混杂因素。如果你想看"移除模块后参数量
     真实减少的效果"，可在 AblationConfig 中将 strict_remove=True，
     此时会真正不创建该模块的参数（见下方说明）。

可消融的模块（开关名 -> 对应图中位置）：
  use_club          : CLUB 互信息解耦损失（min I(He;X_high)+I(Hs;X_low)）
  use_ms_context    : MultiScaleContext，即 "Env Context H'_e"（多尺度环境上下文提炼）
  use_scgmp         : SCGMP，即 "SCGMP Layer H'_s"（空间因果门控消息传递）
  use_rank_loss     : LowRankGCN 的核loss（rank_loss，鼓励 U 列正交，紧凑表征）
  use_wind_mask     : SparseGCN 的风向掩码（仅 SDWPF 数据集有意义）

每个开关关闭后的"退化"行为：
  use_club=False        -> 不计算/不优化 CLUB 损失，He/Hs 不受互信息解耦约束
  use_ms_context=False  -> H'_e 直接取 backbone 输出的 He（池化后的单步特征），
                            不做多尺度时间卷积精炼
  use_scgmp=False        -> H'_s 直接取 backbone 输出的 Hs，不做空间消息传播
  use_rank_loss=False    -> loss 中不加 lambda_rank * rank_loss 项
  use_wind_mask=False    -> SparseGCN 退化为无向自适应图（仅影响 sdwpf 预设）
"""

from dataclasses import dataclass
import torch

from model import (
    GridCFN, DualTrackBackbone, CLUBEstimator,
    MultiScaleContext, SCGMP, CFMVectorField,
)


@dataclass
class AblationConfig:
    use_club:       bool = True
    use_ms_context: bool = True
    use_scgmp:      bool = True
    use_rank_loss:  bool = True
    use_wind_mask:  bool = True
    name:           str  = "full"   # 用于日志/结果命名

    def tag(self) -> str:
        """生成简洁标签，用于文件名和日志区分实验"""
        flags = [
            "club"   if self.use_club       else "noclub",
            "ms"     if self.use_ms_context else "noms",
            "scgmp"  if self.use_scgmp      else "noscgmp",
            "rank"   if self.use_rank_loss  else "norank",
            "wind"   if self.use_wind_mask  else "nowind",
        ]
        return f"{self.name}__" + "_".join(flags)


class AblationGridCFN(GridCFN):
    """
    继承 GridCFN，仅重写 forward / cfm相关入口处的"路由逻辑"。
    所有子模块（backbone, club_e, club_s, ms_context, scgmp, vector_field）
    与父类完全一致，参数量、初始化方式都不变 —— 消融只发生在 forward 计算图里，
    确保"对比的是模块的功能贡献"，而不是"对比模型容量的变化"。
    """

    def __init__(self, *args, ablation: AblationConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation = ablation or AblationConfig()

        # use_wind_mask=False 时，把 backbone 的 wind_mask 屏蔽掉（仅影响 sdwpf）
        if not self.ablation.use_wind_mask and self.backbone.wind_mask is not None:
            # 保留 buffer 本身（避免 state_dict key 缺失报错），forward 时不传入即可
            self._disable_wind_mask = True
        else:
            self._disable_wind_mask = False

    def rank_loss(self) -> torch.Tensor:
        if not self.ablation.use_rank_loss:
            return torch.zeros((), device=self.backbone.gcn_e.U.device)
        return super().rank_loss()

    def forward(self, x, adj_norm, edge_index):
        # ── backbone：若关闭 wind_mask，临时屏蔽后恢复，不破坏原 state_dict ──
        if self._disable_wind_mask:
            saved_mask = self.backbone.wind_mask
            self.backbone.wind_mask = None
            try:
                He_seq, He, Hs, X_low_pooled, X_high_pooled = self.backbone(x, adj_norm=adj_norm)
            finally:
                self.backbone.wind_mask = saved_mask
        else:
            He_seq, He, Hs, X_low_pooled, X_high_pooled = self.backbone(x, adj_norm=adj_norm)

        # ── Env Context (MultiScaleContext) ──
        if self.ablation.use_ms_context:
            He_prime = self.ms_context(He_seq)
        else:
            # 退化：直接用池化后的单步环境特征 He，不做多尺度时间卷积精炼。
            # 注意 ms_context.proj 的输出维度是 ms_out_dim，而 He 的维度是 env_dim，
            # 二者在默认配置里通常相等（env_dim == ms_out_dim == 32），但为了在
            # 维度不一致的配置下也能跑通，这里用一个固定（不参与训练核心对比的）
            # 线性投影对齐维度。该投影层在 __init__ 中按需创建，见下方 _get_passthrough_proj。
            He_prime = self._passthrough_env(He)

        # ── SCGMP Layer ──
        if self.ablation.use_scgmp:
            Hs_prime = self.scgmp(Hs, He, edge_index)
        else:
            # 退化：直接用 backbone 输出的 Hs，不做空间因果消息传播。
            Hs_prime = Hs

        return He_prime, Hs_prime, He, Hs, X_low_pooled, X_high_pooled

    def _passthrough_env(self, He: torch.Tensor) -> torch.Tensor:
        """
        He: [B, N, env_dim] -> 直接返回，不做任何精炼，这才是真正意义上的
        "什么都不做"。

        要求 env_dim == ms_out_dim（当前所有 preset 均满足）。之前这里有一个
        "维度不匹配时用随机正交投影对齐形状"的 fallback，已移除：
          1) 该投影是临时挂在 forward 里的普通属性，不会被 state_dict()/
             parameters()/.to(device) 纳入管理，一旦模型后续被搬到别的
             device 或做 checkpoint 续训，会悄悄出问题；
          2) 更本质的问题是，这种"随机重新混合"并不是真正的直通，会给
             "移除精炼模块"的对比引入一个不可控的随机因素，违背消融实验
             公平性的初衷。如果未来确实需要 env_dim != ms_out_dim 的配置，
             应该在外层显式处理（例如改 ms_out_dim 与 env_dim 一致），
             而不是在这里悄悄补一个形状对齐层。
        """
        env_dim    = He.shape[-1]
        ms_out_dim = self.ms_context.proj.out_features
        assert env_dim == ms_out_dim, (
            f"AblationGridCFN._passthrough_env 要求 env_dim({env_dim}) == "
            f"ms_out_dim({ms_out_dim})，否则 use_ms_context=False 时无法做到"
            f"真正意义上的直通（无额外可学习/随机变换）。请调整模型配置使两者相等。"
        )
        return He

    def cfm_loss(self, He_prime, Hs_prime, y_target, n_t_samples=4, sigma_min=0.01):
        return super().cfm_loss(He_prime, Hs_prime, y_target,
                                 n_t_samples=n_t_samples, sigma_min=sigma_min)


def build_ablation_model(cfg, in_dim, n_nodes, wind_mask, ablation: AblationConfig):
    """
    与 main.py::build_model 等价，但构造的是 AblationGridCFN，
    并把 ablation 配置传进去。模型超参（层数/维度等）与原模型完全一致。
    """
    m = cfg.model
    return AblationGridCFN(
        n_nodes=n_nodes,
        in_dim=in_dim,
        gcn_hidden=m.gcn_hidden, gcn_layers=m.gcn_layers,
        tcn_hidden=m.tcn_hidden, tcn_layers=m.tcn_layers,
        env_dim=m.env_dim, stoch_dim=m.stoch_dim, ms_out_dim=m.ms_out_dim,
        n_scg_layers=m.n_scg_layers, out_dim=m.out_dim, lambda_mi=m.lambda_mi,
        cfm_hidden=getattr(m, "cfm_hidden", 256),
        cfm_time_emb_dim=getattr(m, "cfm_time_emb_dim", 16),
        chunk_size=getattr(m, "chunk_size", 16384),
        ms_dilations=getattr(m, "ms_dilations", (1, 7, 30)),
        T_out=cfg.data.T_out,
        T_in=cfg.data.T_in,
        rank_r=getattr(m, "rank_r", 8),
        lambda_rank=getattr(m, "lambda_rank", 0.01),
        freq_candidates=getattr(m, "freq_candidates", (12, 24, 48, 96)),
        wind_mask=wind_mask,
        ablation=ablation,
    )