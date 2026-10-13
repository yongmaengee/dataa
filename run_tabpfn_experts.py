"""TabPFN 그룹 전문가 (별도 프로세스; XGBoost와 같은 프로세스에서는 libomp 데드락).

target별 확정 표현(PRESET)으로 그룹별(FI/KE/SH/WE) + 전체('all') TabPFN을 두고,
92일 블록마다 블록 시작 - purge 이전 행으로만 fit한 causal OOF 예측확률을 생성 → stacking feature.
test(925~1124)는 train 전체로 fit. 'all' 전문가는 단독 TabPFN 후보로도 사용.

uv run python run_tabpfn_experts.py [--version v3.5|v3.5-fast] [--device mps|cpu] [--n-blocks 8]
→ results/tabpfn_experts.csv (Time_step + EXP_{target}_{group})
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ablation import PRESET, make_features, walk_forward
from src.data import ROOT, TARGETS, group_columns, load
from src.tabpfn_util import fit_predict, load_env

load_env()
p = argparse.ArgumentParser()
p.add_argument("--version", default="v3.5", choices=["v3.5", "v3.5-fast"])
p.add_argument("--device", default="mps")
p.add_argument("--n-blocks", type=int, default=8)
args = p.parse_args()

t0 = time.time()
df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, TARGETS].to_numpy().astype(int)
n = int(is_train.sum())
train_idx, test_idx = np.where(is_train)[0], np.where(~is_train)[0]
blocks = walk_forward(n, n_folds=args.n_blocks)
print("blocks (val_start..val_end Time_step):", [(int(va[0]) + 1, int(va[-1]) + 1) for _, va in blocks], flush=True)

# 전문가 하나가 끝날 때마다 저장 → 중단 시 재실행하면 남은 것만 계산
csv_path, meta_path = ROOT / "results/tabpfn_experts.csv", ROOT / "results/tabpfn_experts.json"
if csv_path.exists():
    out, meta = pd.read_csv(csv_path), json.load(open(meta_path))
    assert (out["Time_step"].to_numpy() == df["Time_step"].to_numpy()).all()
else:
    out, meta = pd.DataFrame({"Time_step": df["Time_step"].to_numpy()}), {"config": vars(args), "block_auc": {}}
for ti, t in enumerate(TARGETS):
    cfg = PRESET[t]
    for g in [*groups, "all"]:
        if f"EXP_{t}_{g}" in out.columns:
            print(f"skip {t} {g} (이미 완료)", flush=True)
            continue
        include = None if g == "all" else [g]
        pred = np.full(len(df), np.nan)
        aucs = []
        for tr, va in blocks:
            fit_mask = np.zeros(len(df), bool)
            fit_mask[tr] = True
            X = make_features(df, fit_mask, groups, **cfg, include=include).to_numpy(np.float32)
            pred[va] = fit_predict(X[tr], y[tr, ti], X[va], args.version, args.device)
            aucs.append(float(roc_auc_score(y[va, ti], pred[va])))
        X = make_features(df, is_train, groups, **cfg, include=include).to_numpy(np.float32)
        pred[test_idx] = fit_predict(X[train_idx], y[:, ti], X[test_idx], args.version, args.device)
        out[f"EXP_{t}_{g}"] = pred
        meta["block_auc"][f"{t}|{g}"] = aucs
        out.to_csv(csv_path, index=False)
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"[{time.time() - t0:5.0f}s] {t} {g:4s} n_feat={X.shape[1]:5d} block AUC={np.round(aucs, 3).tolist()} "
              f"last5 mean={np.mean(aucs[-5:]):.3f}", flush=True)

print(f"saved results/tabpfn_experts.csv ({time.time() - t0:.0f}s)")
