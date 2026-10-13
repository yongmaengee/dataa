"""T1 전용: 윈도우(W일 × F feature) 입력으로 mlp_flat / conv1d / conv2d / conv2d_xattn 비교 (walk-forward 5 fold).

입력: 원본 284 계열을 fold train 통계로 표준화 → 과거 W일 윈도우 → 윈도우별 instance norm(수준 제거, 모양만).
열 순서: 그룹 → 그룹 내 d1 상관 계층군집 leaf 순서 (2D conv의 feature 축 인접성 확보, 비지도).
학습: AdamW, BCE, 입력 가우시안 노이즈, inner tail(마지막 60일, purge 3) early stopping, 3 seed 확률 평균.
참고 기준: xgb[base] 0.580, tabpfn[base] 0.587 (results/final.json)

uv run python run_conv2d.py [--W 20] [--seeds 3] [--quick]
→ results/conv2d_T1.json
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import roc_auc_score

from src.ablation import walk_forward
from src.conv2d import ConvNet, MLPFlat
from src.data import ROOT, group_columns, load
from src.train import windows

p = argparse.ArgumentParser()
p.add_argument("--W", type=int, default=20)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
p.add_argument("--quick", action="store_true")
args = p.parse_args()
CFG = dict(lr=1e-3, wd=1e-2, bs=64, max_epochs=80, patience=12, noise=0.1, drop=0.3)
if args.quick:
    args.seeds, CFG["max_epochs"], CFG["patience"] = 1, 10, 4
MODELS = {
    "mlp_flat": lambda W, F: MLPFlat(W, F, drop=CFG["drop"]),
    "conv1d": lambda W, F: ConvNet(W, F, "conv1d", drop=CFG["drop"]),
    "conv2d": lambda W, F: ConvNet(W, F, "conv2d", drop=CFG["drop"]),
    "conv2d_xattn": lambda W, F: ConvNet(W, F, "conv2d_xattn", drop=CFG["drop"]),
}

df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, "Target_1"].to_numpy().astype(np.float32)
folds = walk_forward(int(is_train.sum()))
dev = args.device


def column_order(tr):
    """그룹 순서 유지, 그룹 내부는 train 행 d1 상관 기반 계층군집 leaf 순서."""
    order = []
    for g, cols in groups.items():
        D = df.loc[tr, cols].diff().iloc[1:]
        D = D.loc[:, D.std() > 0]
        if D.shape[1] > 2:
            C = D.corr().fillna(0).to_numpy()
            Z = linkage(1 - C[np.triu_indices_from(C, 1)], "average")
            order += list(D.columns[leaves_list(Z)])
        else:
            order += list(D.columns)
    return order


def make_input(tr):
    cols = column_order(tr)
    A = df[cols].to_numpy(np.float32)
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
    Xw = windows((A - mu) / sd, args.W)                       # (N, W, F)
    Xw = (Xw - Xw.mean(1, keepdims=True)) / (Xw.std(1, keepdims=True) + 1e-3)  # 윈도우별 instance norm
    return torch.from_numpy(Xw[:, None]).to(dev)              # (N, 1, W, F)


@torch.no_grad()
def predict(model, X, idx):
    model.eval()
    return torch.sigmoid(model(X[torch.as_tensor(idx, device=dev)])).cpu().numpy()


def fit(make, X, tr_in, es, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make(X.shape[2], X.shape[3]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    yt = torch.as_tensor(y, device=dev)
    best, best_state, wait = -1, None, 0
    for ep in range(CFG["max_epochs"]):
        model.train()
        for s in range(0, len(tr_in), CFG["bs"]):
            bi = torch.as_tensor(rng.permutation(tr_in)[s:s + CFG["bs"]], device=dev)
            xb = X[bi] + CFG["noise"] * torch.randn_like(X[bi])
            loss = F_.binary_cross_entropy_with_logits(model(xb), yt[bi])
            opt.zero_grad(), loss.backward(), opt.step()
        auc = roc_auc_score(y[es], predict(model, X, es))
        if auc > best:
            best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        elif (wait := wait + 1) >= CFG["patience"]:
            break
    model.load_state_dict(best_state)
    return model


t0 = time.time()
res = {"config": {**CFG, **vars(args)}, "runs": {}}
for name, make in MODELS.items():
    aucs, gates = [], []
    for k, (tr, va) in enumerate(folds):
        X = make_input(tr)
        tr_in, es = tr[:-63], tr[-60:]
        probs, g = [], []
        for seed in range(args.seeds):
            m = fit(make, X, tr_in, es, seed)
            probs.append(predict(m, X, va))
            if (gv := m.gate() if hasattr(m, "gate") else None) is not None:
                g.append(gv)
        aucs.append(roc_auc_score(y[va], np.mean(probs, 0)))
        gates += g
    r = dict(mean=float(np.mean(aucs)), se=float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))), folds=np.round(aucs, 3).tolist(),
             n_params=sum(p.numel() for p in m.parameters()))
    if gates:
        r["gate_mean"] = float(np.mean(gates))
    res["runs"][name] = r
    print(f"[{time.time() - t0:5.0f}s] {name:13s} AUC={r['mean']:.3f}±{r['se']:.3f} folds={r['folds']} "
          f"params={r['n_params']}" + (f" gate={r['gate_mean']:+.3f}" if gates else ""), flush=True)

json.dump(res, open(ROOT / "results/conv2d_T1.json", "w"), indent=1)
tab = pd.DataFrame(res["runs"]).T
print("\n== T1 walk-forward AUC (참고: xgb[base] 0.580±0.019, tabpfn[base] 0.587±0.013)\n", tab.to_string())
print(f"total {time.time() - t0:.0f}s")
