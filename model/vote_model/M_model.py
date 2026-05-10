from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGRUClassifier(nn.Module):
    """仅文本：6 题 Nanbeige 向量序列 [B,T,D] → GRU → 三分类（计划书阶段四）。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.text_dim = int(args.text_dim)
        self.hidden = int(getattr(args, "gru_hidden_dim", 128))
        self.gru_dropout = float(getattr(args, "gru_dropout", 0.3))
        nl = max(1, int(getattr(args, "gru_num_layers", 1)))
        # PyTorch GRU：仅 num_layers>1 时 inter-layer dropout 有效；单层用输出后 Dropout
        self.gru = nn.GRU(
            self.text_dim,
            self.hidden,
            num_layers=nl,
            batch_first=True,
            dropout=float(getattr(args, "gru_inter_dropout", 0.0)) if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(self.gru_dropout)
        self.fc = nn.Linear(self.hidden, int(args.target_dim))
        self.pool = str(getattr(args, "text_gru_pool", "last")).lower()

    def forward(self, audio_feat, video_feat, text_feat, hand_feat):
        del audio_feat, video_feat, hand_feat
        if getattr(self.args, "freeze_text_features", False):
            text_feat = text_feat.detach()
        out, hn = self.gru(text_feat)
        if self.pool == "mean":
            x = out.mean(dim=1)
        else:
            x = hn[-1]
        return self.fc(self.drop(x))


class TextOnlyMLPClassifier(nn.Module):
    """仅文本：将 [B,T,D] 展平后过 MLP（单模态基线）。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        nq = len(args.question)
        d = int(args.text_dim)
        fused = nq * d
        drop = float(getattr(args, "text_mlp_dropout", 0.3))
        hid = int(getattr(args, "text_mlp_hidden", 512))
        self.net = nn.Sequential(
            nn.Linear(fused, hid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid, int(args.target_dim)),
        )

    def forward(self, audio_feat, video_feat, text_feat, hand_feat):
        del audio_feat, video_feat, hand_feat
        if getattr(self.args, "freeze_text_features", False):
            text_feat = text_feat.detach()
        B, T, D = text_feat.shape
        x = text_feat.reshape(B, T * D)
        return self.net(x)


class AudioTextMLPClassifier(nn.Module):
    """仅音频+文本：adapter → base_dim，拼接 2*base_dim，题维 mean，MLP 头（无视频/手工/跨模态）。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.base_dim = int(getattr(args, "base_dim", 768))
        self.at_dim = self.base_dim * 2
        mlp_drop = float(getattr(args, "mlp_dropout", 0.25))
        hid = int(getattr(args, "at_mlp_hidden", 512))

        self.audio_adapter = nn.Sequential(
            nn.Linear(int(args.audio_dim), self.base_dim * 3),
            nn.GELU(),
        )
        self.text_adapter = nn.Sequential(
            nn.Linear(int(args.text_dim), self.base_dim * 3),
            nn.GELU(),
        )
        self.feat_proj = nn.Linear(self.base_dim * 3, self.base_dim)
        self.at_ln = (
            nn.LayerNorm(self.at_dim)
            if bool(getattr(args, "fused_layer_norm", False))
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Linear(self.at_dim, hid),
            nn.ReLU(),
            nn.Dropout(mlp_drop),
            nn.Linear(hid, int(args.target_dim)),
        )

    def forward(self, audio_feat, video_feat, text_feat, hand_feat):
        del video_feat, hand_feat
        if getattr(self.args, "freeze_text_features", False):
            text_feat = text_feat.detach()
            audio_feat = audio_feat.detach()
        B, T, _ = audio_feat.shape
        a = audio_feat.reshape(B * T, -1)
        t = text_feat.reshape(B * T, -1)
        a = self.feat_proj(self.audio_adapter(a))
        t = self.feat_proj(self.text_adapter(t))
        x = torch.cat([a, t], dim=-1)
        x = self.at_ln(x)
        x = x.view(B, T, -1).mean(dim=1)
        return self.head(x)


class _TextEnhancerTransformer(nn.Module):
    """(B*T, D) → view (B,T,D) → 1-layer TransformerEncoder → 扁平化。"""

    def __init__(self, base_dim: int, dim_feedforward: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=base_dim,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        z = x.view(B, T, -1)
        z = self.encoder(z)
        return z.reshape(B * T, -1)


class _TextEnhancerMLP(nn.Module):
    """带残差的 2 层 MLP，逐 (B*T,) 样本作用。"""

    def __init__(self, base_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, base_dim),
        )

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        del B, T
        return x + self.branch(x)


class SharedMLPwEnsemble(torch.nn.Module):
    def __init__(self, args):
        super(SharedMLPwEnsemble, self).__init__()
        self.args = args
        self.base_dim = int(getattr(args, "base_dim", 768))
        self.hand_dim = int(getattr(args, "hand_dim", 4))
        self.fused_dim = self.base_dim * 3 + self.hand_dim
        self.simple_clf = bool(getattr(args, "simple_clf", False))
        self.num_heads = int(getattr(args, "num_heads", 32))
        self.mlp_bottleneck_dim = int(getattr(args, "mlp_bottleneck_dim", 32))

        self.feature_projection = torch.nn.Linear(self.base_dim * 3, self.base_dim)

        self.hand_bn = torch.nn.BatchNorm1d(self.hand_dim)

        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.video_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        self.audio_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.audio_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        self.text_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.text_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        mlp_drop = float(getattr(args, "mlp_dropout", 0.0))

        if self.simple_clf:
            self.ensemble = None
            self.simple_head = torch.nn.Linear(self.fused_dim, int(args.target_dim))
        else:

            def _branch():
                return torch.nn.Sequential(
                    torch.nn.Linear(self.fused_dim, self.base_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(mlp_drop),
                    torch.nn.Linear(self.base_dim, self.mlp_bottleneck_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(mlp_drop),
                    torch.nn.Linear(self.mlp_bottleneck_dim, int(args.target_dim)),
                )

            self.ensemble = torch.nn.ModuleList([_branch() for _ in range(self.num_heads)])
            self.simple_head = None

        self.use_temporal_gru = getattr(args, "temporal_gru", False)
        self.temporal_pool = getattr(args, "temporal_pool", "mean")
        self.temporal_bidirectional = bool(
            getattr(args, "temporal_bidirectional", False)
        )
        self.temporal_attn_pool = bool(getattr(args, "temporal_attn_pool", False))
        tdrop = float(getattr(args, "temporal_dropout", 0.1))
        qdim = int(getattr(args, "question_pos_embed_dim", 0) or 0)
        self.temporal_step_dropout_p = float(
            getattr(args, "temporal_step_dropout_p", 0.0) or 0.0
        )
        if qdim > 0:
            nq = max(1, len(getattr(args, "question", [])))
            self.question_embed = nn.Embedding(nq, qdim)
            self.question_fuse = nn.Linear(self.fused_dim + qdim, self.fused_dim)
        else:
            self.question_embed = None
            self.question_fuse = None

        if self.use_temporal_gru:
            self.temporal_dropout = torch.nn.Dropout(tdrop)
            bi = self.temporal_bidirectional
            self.temporal_gru = torch.nn.GRU(
                self.fused_dim,
                self.fused_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=bi,
            )
            gru_out_dim = 2 * self.fused_dim if bi else self.fused_dim
            self.temporal_bi_proj = (
                nn.Linear(gru_out_dim, self.fused_dim) if bi else None
            )
            self.temporal_attn = (
                nn.Linear(self.fused_dim, 1) if self.temporal_attn_pool else None
            )
        else:
            self.temporal_dropout = None
            self.temporal_gru = None
            self.temporal_bi_proj = None
            self.temporal_attn = None

        self.modality_dropout_p = float(getattr(args, "modality_dropout_p", 0.0))

        self.use_cross_modal_attn = bool(getattr(args, "cross_modal_attn", False))
        self.modality_transformer = None
        if self.use_cross_modal_attn:
            nhead = int(getattr(args, "cross_modal_nhead", 8))
            if self.base_dim % nhead != 0:
                raise ValueError(
                    f"base_dim={self.base_dim} 必须能被 cross_modal_nhead={nhead} 整除"
                )
            nlayers = max(1, int(getattr(args, "cross_modal_layers", 1)))
            cmdrop = float(getattr(args, "cross_modal_dropout", 0.1))
            ff_mult = int(getattr(args, "cross_modal_ff_mult", 4))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.base_dim,
                nhead=nhead,
                dim_feedforward=self.base_dim * ff_mult,
                dropout=cmdrop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.modality_transformer = nn.TransformerEncoder(
                enc_layer, num_layers=nlayers
            )

        self.use_head_weights = bool(getattr(args, "head_weights", False))
        self.use_time_weights = bool(getattr(args, "time_weights", False))
        nq = max(1, len(getattr(args, "question", ["q1"])))
        self.num_question_steps = nq

        if not self.simple_clf and self.ensemble is not None and self.use_head_weights:
            self.head_logits = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.register_parameter("head_logits", None)

        if self.use_time_weights and not self.use_temporal_gru:
            self.time_logits = nn.Parameter(torch.zeros(nq))
        else:
            self.register_parameter("time_logits", None)

        te = str(getattr(args, "text_enhancer", "none") or "none").lower()
        enh_dim = int(getattr(args, "text_enhancer_dim", 512))
        te_drop = float(getattr(args, "mlp_dropout", 0.0))
        self.text_enhancer: nn.Module | None = None
        if te == "transformer":
            if self.base_dim % 4 != 0:
                raise ValueError(
                    f"text_enhancer=transformer 需要 base_dim={self.base_dim} 能被 4 整除"
                )
            self.text_enhancer = _TextEnhancerTransformer(
                self.base_dim, enh_dim, te_drop
            )
        elif te == "mlp":
            self.text_enhancer = _TextEnhancerMLP(self.base_dim, enh_dim, te_drop)
        elif te not in ("none", ""):
            raise ValueError(f"未知 text_enhancer={te!r}，请用 none|transformer|mlp")

        self.fused_ln = (
            nn.LayerNorm(self.fused_dim)
            if bool(getattr(args, "fused_layer_norm", False))
            else nn.Identity()
        )

    def _combine_head_outputs(self, stacked: torch.Tensor) -> torch.Tensor:
        """stacked: (H, N, C) → (N, C)。"""
        if self.head_logits is None:
            return stacked.mean(dim=0)
        w = F.softmax(self.head_logits, dim=0)
        return (stacked * w.view(-1, 1, 1)).sum(dim=0)

    def encode_multimodal_chunk(self, audio_feat, video_feat, text_feat, hand_feat):
        B, T, _ = video_feat.shape
        video_feat = video_feat.reshape(B * T, -1)
        text_feat = text_feat.reshape(B * T, -1)
        audio_feat = audio_feat.reshape(B * T, -1)
        video_feat = self.feature_projection(self.video_adapter(video_feat))
        audio_feat = self.feature_projection(self.audio_adapter(audio_feat))
        text_feat = self.feature_projection(self.text_adapter(text_feat))
        if self.text_enhancer is not None:
            text_feat = self.text_enhancer(text_feat, B, T)

        if self.training and self.modality_dropout_p > 0:
            vb = video_feat.view(B, T, -1)
            ab = audio_feat.view(B, T, -1)
            tb = text_feat.view(B, T, -1)
            device = vb.device
            drop = torch.rand(B, device=device) < self.modality_dropout_p
            choice = torch.randint(0, 3, (B,), device=device)
            d = drop.float().unsqueeze(-1).unsqueeze(-1)
            vm = (choice == 0).float().unsqueeze(-1).unsqueeze(-1)
            tm = (choice == 1).float().unsqueeze(-1).unsqueeze(-1)
            am = (choice == 2).float().unsqueeze(-1).unsqueeze(-1)
            vb = vb * (1.0 - d * vm)
            tb = tb * (1.0 - d * tm)
            ab = ab * (1.0 - d * am)
            video_feat = vb.reshape(B * T, -1)
            audio_feat = ab.reshape(B * T, -1)
            text_feat = tb.reshape(B * T, -1)

        if self.modality_transformer is not None:
            tokens = torch.stack([video_feat, text_feat, audio_feat], dim=1)
            tokens = self.modality_transformer(tokens)
            video_feat, text_feat, audio_feat = (
                tokens[:, 0],
                tokens[:, 1],
                tokens[:, 2],
            )

        multi_modal_chunk = torch.cat([video_feat, text_feat, audio_feat], dim=-1)
        h = hand_feat.reshape(B * T, self.hand_dim)
        h = self.hand_bn(h)
        multi_modal_chunk = torch.cat([multi_modal_chunk, h], dim=-1)
        return multi_modal_chunk, B, T

    def forward_heads(self, multi_modal_chunk, B, T):
        x = multi_modal_chunk.view(B, T, -1)
        if self.question_fuse is not None:
            idx = torch.arange(T, device=x.device, dtype=torch.long)
            qe = self.question_embed(idx)
            x = torch.cat([x, qe.unsqueeze(0).expand(B, -1, -1)], dim=-1)
            x = self.question_fuse(x)
        if self.training and self.temporal_step_dropout_p > 0:
            p = self.temporal_step_dropout_p
            mask = torch.bernoulli(
                torch.full((B, T, 1), 1.0 - p, device=x.device, dtype=x.dtype)
            )
            x = x * mask / (1.0 - p + 1e-8)
        multi_modal_chunk = x.reshape(B * T, -1)
        multi_modal_chunk = self.fused_ln(multi_modal_chunk)

        if self.simple_clf:
            seq_repr = multi_modal_chunk.view(B, T, -1).mean(dim=1)
            return self.simple_head(seq_repr)

        if self.use_temporal_gru:
            x = multi_modal_chunk.view(B, T, -1)
            x = self.temporal_dropout(x)
            out, _ = self.temporal_gru(x)
            if self.temporal_bi_proj is not None:
                out = self.temporal_bi_proj(out)
            if self.temporal_attn is not None:
                w = torch.softmax(self.temporal_attn(out), dim=1)
                seq_repr = (out * w).sum(dim=1)
            elif self.temporal_pool == "last":
                seq_repr = out[:, -1, :]
            else:
                seq_repr = out.mean(dim=1)
            seq_repr = self.temporal_dropout(seq_repr)
            outputs = torch.stack([mlp(seq_repr) for mlp in self.ensemble], dim=0)
            return self._combine_head_outputs(outputs)
        outputs = torch.stack([mlp(multi_modal_chunk) for mlp in self.ensemble], dim=0)
        logits = self._combine_head_outputs(outputs)
        logits_bt = logits.view(B, T, -1)
        if self.time_logits is not None:
            if T != self.time_logits.shape[0]:
                raise RuntimeError(
                    f"time_weights: batch T={T} 与参数长度 {self.time_logits.shape[0]} 不一致"
                )
            wt = F.softmax(self.time_logits, dim=0)
            return (logits_bt * wt.view(1, T, 1)).sum(dim=1)
        return logits_bt.mean(dim=1)

    def forward(self, audio_feat, video_feat, text_feat, hand_feat):
        multi_modal_chunk, B, T = self.encode_multimodal_chunk(
            audio_feat, video_feat, text_feat, hand_feat
        )
        return self.forward_heads(multi_modal_chunk, B, T)
