"""Group-wise gated cross-attention 모델과 concat-MLP 대조군.

입력 batch: dict 스트림명 -> (B, W, F). 'Q'는 query 스트림, 나머지는 KV 그룹.
각 스트림의 시점별 벡터에는 calendar(sin/cos 요일)가 붙어 있어 시점 정렬 단서로 쓰임.
"""
import math

import torch
import torch.nn as nn


class GroupEncoder(nn.Module):
    """그룹별 독립 encoder: 투영 + 시점 위치 임베딩 + self-attention 1층."""

    def __init__(self, f_in, d, W, heads, dropout):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        layer.self_attn.dropout = 0.0  # MPS SDPA는 attention dropout 미지원 → attention 가중치 dropout만 끔
        self.enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self, x):
        return self.enc(self.inp(x) + self.pos)


class GatedCrossAttn(nn.Module):
    """x <- x + tanh(gate) * Attn(LN(x), LN(kv)). gate=0 초기화로 처음엔 항등 함수."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)  # MPS 제약, 위와 동일
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, kv):
        k = self.nkv(kv)
        h, w = self.attn(self.nq(x), k, k, need_weights=True, average_attn_weights=True)
        return x + torch.tanh(self.gate) * h, w


class FFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):
        return x + self.f(x)


def ffn_depth(W, kv_dims, C, d_max=4):
    """FFN depth를 KV 크기(토큰 수 x 입력 차원 = W * sum F_kv)에 비례하게 설정."""
    return int(min(max(math.ceil(W * sum(kv_dims) / C), 1), d_max))


class XAttnNet(nn.Module):
    def __init__(self, dims, kv_groups, W, d=32, heads=4, dropout=0.2, ffn_C=2000, use_cross=True):
        super().__init__()
        self.kv_groups, self.use_cross = list(kv_groups), use_cross
        self.enc = nn.ModuleDict({g: GroupEncoder(dims[g], d, W, heads, dropout) for g in ["Q", *kv_groups]})
        self.gemb = nn.ParameterDict({g: nn.Parameter(torch.randn(1, 1, d) * 0.02) for g in kv_groups})
        self.xattn = GatedCrossAttn(d, heads, dropout)
        self.depth = ffn_depth(W, [dims[g] for g in kv_groups], ffn_C)
        self.ffn = nn.ModuleList([FFN(d, dropout) for _ in range(self.depth)])
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(dropout), nn.Linear(2 * d, 1))

    def forward(self, batch, kv_mode="normal"):
        """kv_mode: 'normal' | 'shuffle'(batch 내 KV를 다른 샘플 것으로 교체) | 'zero'(cross-attn 생략)."""
        x = self.enc["Q"](batch["Q"])
        w = None
        if self.use_cross and kv_mode != "zero":
            kv = torch.cat([self.enc[g](batch[g]) + self.gemb[g] for g in self.kv_groups], dim=1)
            if kv_mode == "shuffle":
                kv = kv[torch.randperm(kv.shape[0], device=kv.device)]
            x, w = self.xattn(x, kv)
        for f in self.ffn:
            x = f(x)
        z = torch.cat([x[:, -1], x.mean(1)], dim=-1)  # 가장 최근 시점 토큰 + 윈도우 평균
        return self.head(z).squeeze(-1), w


class ConcatMLP(nn.Module):
    """대조군: 당일 시점의 모든 스트림 feature를 이어 붙인 MLP (attention 없음)."""

    def __init__(self, dims, kv_groups, dropout=0.2, **_):
        super().__init__()
        self.groups = ["Q", *kv_groups]
        f = sum(dims[g] for g in self.groups)
        self.net = nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, batch, kv_mode="normal"):
        return self.net(torch.cat([batch[g][:, -1] for g in self.groups], dim=-1)).squeeze(-1), None
