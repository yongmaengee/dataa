"""T1 전용: 7일 청크 출력 + 학습 가능한 선택기 (walk-forward 5 fold).

입력: 확정 표현(change+zscore) t 시점 feature → 공유 MLP 인코더 → head K개, 각 head는 t..t+H-1 label H개의 logit(청크).
선택 학습:
  mcl    : multiple-choice learning. 샘플마다 BCE가 최소인 head만 학습(oracle loss) + selector가 그 head 번호를 분류(CE).
           추론: selector argmax head의 청크. 'oracle'은 정답 head를 알 때의 상한(참고).
  gate   : gate 가중합 청크에 BCE (soft 선택).
  single : head 1개 (단순 multi-horizon, 대조군).
일별 점수: 날짜 d는 offset k=0..H-1의 청크(윈도우 끝 d-k의 k번째 출력)에서 H개 예측을 받음 → 평균. k=0만 쓴 값도 보고.

uv run python run_chunk.py [--K 4] [--H 7] [--seeds 3] [--quick]
→ results/chunk_T1.json
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.ablation import PRESET, make_features, walk_forward
from src.data import ROOT, group_columns, load

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=4)
p.add_argument("--H", type=int, default=7)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
p.add_argument("--quick", action="store_true")
p.add_argument("--enc", nargs="*", default=["mlp", "conv1d"], help="인코더: mlp(당일 feature) / conv1d(과거 W일 윈도우, 시간축 conv + FFN)")
p.add_argument("--modes", nargs="*", default=["single", "mcl"])
p.add_argument("--W", type=int, default=10)
p.add_argument("--drop", type=float, default=0.4)
p.add_argument("--sel-w", type=float, default=1.0)
args = p.parse_args()
CFG = dict(lr=1e-3, wd=2e-2, bs=64, max_epochs=100, patience=15, noise=0.15, feat_drop=0.2, step_drop=0.2,
           drop=args.drop, d=64, sel_w=args.sel_w)
if args.quick:
    args.seeds, CFG["max_epochs"], CFG["patience"] = 1, 10, 4
dev, H, K = args.device, args.H, args.K

df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, "Target_1"].to_numpy().astype(np.float32)
n = len(y)
folds = walk_forward(n)


def chunk_targets(idx):
    """윈도우 끝 t에 대한 target 청크 y[t..t+H-1]. 범위 밖(미래 label 없음)은 mask 0."""
    T = np.zeros((len(idx), H), np.float32)
    M = np.zeros((len(idx), H), np.float32)
    for j, t in enumerate(idx):
        for k in range(H):
            if t + k < n:
                T[j, k], M[j, k] = y[t + k], 1.0
    return torch.as_tensor(T, device=dev), torch.as_tensor(M, device=dev)


class Conv1DEnc(nn.Module):
    """(B, W, F) → 시점별 투영 → 시간축 conv ×2 → [마지막 시점, 평균] → d."""

    def __init__(self, f, d, drop):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(f, d), nn.GELU())
        self.conv = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GELU(), nn.Dropout(drop), nn.Conv1d(d, d, 3, padding=1), nn.GELU())
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        h = self.conv(self.proj(x).transpose(1, 2)).transpose(1, 2)  # (B, W, d)
        return self.out(torch.cat([h[:, -1], h.mean(1)], -1))


class ChunkNet(nn.Module):
    def __init__(self, f, mode, enc="mlp"):
        super().__init__()
        d, drop = CFG["d"], CFG["drop"]
        self.mode, self.K = mode, (1 if mode == "single" else K)
        if enc == "conv1d":
            self.enc = Conv1DEnc(f, d, drop)
        else:
            self.enc = nn.Sequential(nn.Dropout(drop), nn.Linear(f, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d), nn.GELU())
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.heads = nn.Linear(d, self.K * H)
        self.sel = nn.Sequential(nn.Dropout(drop), nn.Linear(d, self.K)) if mode in ("mcl", "gate") else None

    def forward(self, x):
        z = self.enc(x)
        z = z + self.ffn(z)
        logits = self.heads(z).view(-1, self.K, H)  # (B, K, H)
        sel = self.sel(z) if self.sel is not None else None  # (B, K)
        return logits, sel


def loss_fn(model, logits, sel, T, M):
    bce = F.binary_cross_entropy_with_logits(logits, T[:, None, :].expand_as(logits), reduction="none")  # (B,K,H)
    per_head = (bce * M[:, None, :]).sum(-1) / M.sum(-1, keepdim=True).clamp_min(1)  # (B,K)
    if model.mode == "single":
        return per_head[:, 0].mean()
    if model.mode == "gate":
        w = torch.softmax(sel, -1)  # soft 선택: 가중합 청크에 BCE
        mix = (w[:, :, None] * torch.sigmoid(logits)).sum(1).clamp(1e-6, 1 - 1e-6)
        return (F.binary_cross_entropy(mix, T, reduction="none") * M).sum() / M.sum()
    best = per_head.argmin(1)  # mcl: oracle loss + selector CE
    oracle = per_head.gather(1, best[:, None]).mean()
    return oracle + CFG["sel_w"] * F.cross_entropy(sel, best)


@torch.no_grad()
def predict_chunks(model, X, idx):
    """반환 dict: 방식 -> (len(idx), H) 확률. mcl은 selected/oracle(정답 head, 참고)/mean, gate는 mixture/mean."""
    model.eval()
    logits, sel = model(X[torch.as_tensor(idx, device=dev)])
    P = torch.sigmoid(logits)
    out = {"mean": P.mean(1)}
    if model.mode == "single":
        out = {"single": P[:, 0]}
    elif model.mode == "gate":
        out["mixture"] = (torch.softmax(sel, -1)[:, :, None] * P).sum(1)
    else:
        out["selected"] = P[torch.arange(len(idx)), sel.argmax(1)]
        T, M = chunk_targets(idx)
        bce = F.binary_cross_entropy_with_logits(logits, T[:, None, :].expand_as(logits), reduction="none")
        best = ((bce * M[:, None, :]).sum(-1)).argmin(1)
        out["oracle"] = P[torch.arange(len(idx)), best]
    return {k: v.cpu().numpy() for k, v in out.items()}


def daily_scores(chunks, idx_all, va):
    """chunks: (len(idx_all), H) — 윈도우 끝 idx_all[j]의 청크. 날짜 d의 점수 = 평균_k chunks[pos(d-k), k]. 반환: (k0, mean_k) for va."""
    pos = {t: j for j, t in enumerate(idx_all)}
    k0, mk = np.zeros(len(va)), np.zeros(len(va))
    for i, d in enumerate(va):
        vals = [chunks[pos[d - k], k] for k in range(H) if (d - k) in pos]
        k0[i], mk[i] = chunks[pos[d], 0], np.mean(vals)
    return k0, mk


def augment(x):
    x = x + CFG["noise"] * torch.randn_like(x)
    x = x * (torch.rand(x.shape[0], *([1] * (x.dim() - 2)), x.shape[-1], device=x.device) > CFG["feat_drop"])
    if x.dim() == 3:  # 윈도우 입력: 시점 dropout (마지막 시점은 유지)
        m = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > CFG["step_drop"]).float()
        m[:, -1] = 1
        x = x * m
    return x


def fit(mode, X, tr_in, es, seed, enc):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ChunkNet(X.shape[-1], mode, enc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    T_all, M_all = chunk_targets(np.arange(n))
    best, best_state, wait = -1, None, 0
    for ep in range(CFG["max_epochs"]):
        model.train()
        for s in range(0, len(tr_in), CFG["bs"]):
            bi = rng.permutation(tr_in)[s:s + CFG["bs"]]
            bt = torch.as_tensor(bi, device=dev)
            xb = augment(X[bt])
            loss = loss_fn(model, *model(xb), T_all[bt], M_all[bt])
            opt.zero_grad(), loss.backward(), opt.step()
        key = {"single": "single", "gate": "mixture", "mcl": "selected"}[mode]
        auc = roc_auc_score(y[es], predict_chunks(model, X, es)[key][:, 0])
        if auc > best:
            best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        elif (wait := wait + 1) >= CFG["patience"]:
            break
    model.load_state_dict(best_state)
    return model


t0 = time.time()
res = {"config": {**CFG, **vars(args)}, "runs": {}}
from src.train import windows  # noqa: E402

for enc in args.enc:
  for mode in args.modes:
    acc = {}
    for tr, va in folds:
        fit_mask = np.zeros(len(df), bool)
        fit_mask[tr] = True
        Xdf = make_features(df, fit_mask, groups, **PRESET["Target_1"]).fillna(0)
        A = Xdf.to_numpy(np.float32)[:n]
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        Z = np.clip((A - mu) / sd, -5, 5)
        X = torch.as_tensor(windows(Z, args.W) if enc == "conv1d" else Z, device=dev)
        # 청크 target이 t+H-1까지 필요 → 학습 행은 val 시작 - purge - (H-1) 이전까지로 당겨 미래 label 누수 방지
        tr_c = tr[tr < va[0] - 3 - (H - 1)]
        tr_in, es = tr_c[:-63], tr_c[-60:]
        idx_all = np.arange(va[0] - H + 1, va[-1] + 1)  # va 날짜가 받을 청크의 윈도우 끝 범위
        preds = {}
        for seed in range(args.seeds):
            m = fit(mode, X, tr_in, es, seed, enc)
            for k, ch in predict_chunks(m, X, idx_all).items():
                preds.setdefault(k, []).append(ch)
        for k, lst in preds.items():
            k0, mk = daily_scores(np.mean(lst, 0), idx_all, va)
            acc.setdefault(f"{k}/k0", []).append(roc_auc_score(y[va], k0))
            acc.setdefault(f"{k}/offset-mean", []).append(roc_auc_score(y[va], mk))
    for k, a in acc.items():
        res["runs"][f"{enc}:{mode}:{k}"] = dict(mean=float(np.mean(a)), se=float(np.std(a, ddof=1) / np.sqrt(len(a))), folds=np.round(a, 3).tolist())
        print(f"[{time.time() - t0:5.0f}s] {enc:6s} {mode:6s} {k:20s} AUC={np.mean(a):.3f}±{np.std(a, ddof=1) / np.sqrt(len(a)):.3f} {np.round(a, 3).tolist()}", flush=True)

json.dump(res, open(ROOT / "results/chunk_T1.json", "w"), indent=1)
print("\n== T1 (참고: tabpfn[base] 0.587±0.013, xgb[base] 0.580±0.019)\n", pd.DataFrame(res["runs"]).T[["mean", "se"]].round(3).to_string())
print(f"total {time.time() - t0:.0f}s")
