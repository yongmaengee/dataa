"""Lead feature 입력 신경망 (T1). 공통: pre-LayerNorm + residual + dropout.

ResMLP   : 당일 feature 벡터 (fi_al 92) → residual MLP 블록 ×L
ResConv1D: 과거 W일 윈도우 (W, F) → 시점별 투영 → residual 시간축 conv 블록 ×L → [마지막, 평균] → FFN
LeadXAttn: Q = FI lead 스트림 윈도우 토큰, KV = KE/SH/WE zscore 스트림 토큰 → self-attn → gated cross-attn → FFN
"""
import torch
import torch.nn as nn

from .xattn import FFN, GatedCrossAttn


class ResBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d), nn.Dropout(drop))

    def forward(self, x):
        return x + self.f(x)


class Head(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.h = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, z):
        return self.h(z).squeeze(-1)


class ResMLP(nn.Module):
    def __init__(self, dims, d=64, L=2, drop=0.3, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(dims["Q"], d), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(d, drop) for _ in range(L)])
        self.head = Head(d, drop)

    def forward(self, b):
        return self.head(self.blocks(self.inp(b["Q"][:, -1])))  # 윈도우의 마지막 시점 = 당일


class ConvBlock(nn.Module):
    """pre-LN residual 시간축 conv (B, W, d)."""

    def __init__(self, d, drop, k=3):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.conv = nn.Sequential(nn.Conv1d(d, d, k, padding=k // 2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(d, d, k, padding=k // 2), nn.Dropout(drop))

    def forward(self, x):
        return x + self.conv(self.ln(x).transpose(1, 2)).transpose(1, 2)


class ResConv1D(nn.Module):
    def __init__(self, dims, d=64, L=2, drop=0.3, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(dims["Q"], d), nn.GELU())
        self.blocks = nn.Sequential(*[ConvBlock(d, drop) for _ in range(L)])
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), ResBlock(d, drop))
        self.head = Head(d, drop)

    def forward(self, b):
        h = self.blocks(self.inp(b["Q"]))
        return self.head(self.out(torch.cat([h[:, -1], h.mean(1)], -1)))


class LeadXAttn(nn.Module):
    def __init__(self, dims, d=64, L=2, drop=0.3, heads=4, kv_groups=("KE", "SH", "WE"), W=10, use_cross=True, **_):
        super().__init__()
        self.kv_groups, self.use_cross = list(kv_groups), use_cross
        self.enc = nn.ModuleDict({g: nn.Sequential(nn.Dropout(drop), nn.Linear(dims[g], d), nn.LayerNorm(d)) for g in ["Q", *kv_groups]})
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.gemb = nn.ParameterDict({g: nn.Parameter(torch.randn(1, 1, d) * 0.02) for g in kv_groups})
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        layer.self_attn.dropout = 0.0  # MPS SDPA 제약
        self.self_attn = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.xattn = GatedCrossAttn(d, heads, 0.0)
        self.ffn = nn.Sequential(*[FFN(d, drop) for _ in range(L)])
        self.head = Head(2 * d, drop)

    def forward(self, b):
        x = self.self_attn(self.enc["Q"](b["Q"]) + self.pos)
        self.last_w = None
        if self.use_cross:
            kv = torch.cat([self.enc[g](b[g]) + self.pos + self.gemb[g] for g in self.kv_groups], 1)
            x, self.last_w = self.xattn(x, kv)
        x = self.ffn(x)
        return self.head(torch.cat([x[:, -1], x.mean(1)], -1))

    def gate(self):
        return float(torch.tanh(self.xattn.gate)) if self.use_cross else None


MODELS = {"mlp": ResMLP, "conv1d": ResConv1D, "xattn": LeadXAttn,
          "xattn_nocross": lambda dims, **kw: LeadXAttn(dims, use_cross=False, **kw)}
