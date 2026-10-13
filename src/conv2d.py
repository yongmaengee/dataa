"""T1용 윈도우 입력 (B, 1, W, F) 모델: 시간축 1D conv / 2D conv / 2D conv + cross-attention + FFN.

행 = 시점(과거 W일), 열 = feature. 2D conv는 feature 축 인접성을 가정하므로 열은 그룹별·상관 군집 순서로 정렬해서 넣는다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_

from .xattn import FFN, GatedCrossAttn


class MLPFlat(nn.Module):
    """대조군: 윈도우를 펼쳐서 MLP (weight sharing 없음)."""

    def __init__(self, W, F, d=32, drop=0.3, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(W * F, d), nn.GELU(), FFN(d, drop),
                                 nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConvNet(nn.Module):
    """kind: 'conv1d'(시간축 커널만, feature 섞지 않음) | 'conv2d' | 'conv2d_xattn'.

    conv2d_xattn: conv 특징맵 (C, Tp, Fp)에서 Q = 시점 토큰(Tp개), KV = feature 토큰(Fp개)으로 gated cross-attention → FFN.
    """

    def __init__(self, W, F, kind, ch=16, d=32, heads=4, drop=0.3, tp=4, fp=16):
        super().__init__()
        self.kind = kind
        kf, sf = (1, 1) if kind == "conv1d" else (5, 2)
        pad = (1, kf // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, (3, kf), padding=pad), nn.GELU(),
            nn.Conv2d(ch, ch, (3, kf), stride=(2, sf), padding=pad), nn.GELU(),
            nn.Conv2d(ch, ch, (3, kf), stride=(2, sf), padding=pad), nn.GELU(),
        )
        self.tp, self.fp = tp, fp
        if kind == "conv2d_xattn":
            self.q, self.kv = nn.Linear(ch * fp, d), nn.Linear(ch * tp, d)
            self.attn = GatedCrossAttn(d, heads, 0.0)
        else:
            self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(ch * tp * fp, d), nn.GELU())
        self.ffn = FFN(d, drop)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x):
        h = self.conv(x)
        # MPS adaptive pool은 크기가 나누어떨어져야 함 → 뒤쪽을 잘라 맞춘 뒤 고정 평균 풀링
        T, F = h.shape[2] // self.tp * self.tp, h.shape[3] // self.fp * self.fp
        h = F_.avg_pool2d(h[:, :, :T, :F], (T // self.tp, F // self.fp))  # (B, C, Tp, Fp)
        if self.kind == "conv2d_xattn":
            q = self.q(h.permute(0, 2, 1, 3).flatten(2))   # (B, Tp, C*Fp) 시점 토큰
            kv = self.kv(h.permute(0, 3, 1, 2).flatten(2))  # (B, Fp, C*Tp) feature 토큰
            z, self.last_attn = self.attn(q, kv)
            z = self.ffn(z).mean(1)
        else:
            z = self.ffn(self.proj(h))
        return self.head(z).squeeze(-1)

    def gate(self):
        return float(torch.tanh(self.attn.gate)) if self.kind == "conv2d_xattn" else None
