"""서브셋/전처리 ablation. 모델(XGBoost 고정 하이퍼파라미터)과 평가(walk-forward 5 fold)는 고정.

uv run python run_ablation.py               # stage 1(표현×축소) → 2(그룹 포함) → 3(EXAONE)
판정: 기준 설정 대비 fold별 paired AUC 차이의 평균 ± 표준오차.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.ablation import REPRS, make_features, walk_forward
from src.data import ROOT, TARGETS, group_columns, load

XGB = dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5,
           min_child_weight=5, reg_lambda=1.0, n_jobs=8)
SEEDS = 3

df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, TARGETS].to_numpy().astype(int)
n = int(is_train.sum())
folds = walk_forward(n)
exaone = pd.read_csv(ROOT / "results/exaone_feats.csv").drop(columns="Time_step")
t0 = time.time()
cache = {}


def evaluate(name, **cfg):
    """fold마다 feature를 다시 만들고(축소 통계는 fold train으로만) XGB를 학습. 반환: (2 targets, n_folds) AUC."""
    key = json.dumps({k: len(v) if k == "extras" else v for k, v in cfg.items()}, sort_keys=True)
    if key in cache:
        return cache[key]
    auc = np.zeros((len(TARGETS), len(folds)))
    n_feat = 0
    for k, (tr, va) in enumerate(folds):
        fit_mask = np.zeros(len(df), bool)
        fit_mask[tr] = True
        X = make_features(df, fit_mask, groups, **cfg).to_numpy(np.float32)
        n_feat = X.shape[1]
        for ti in range(len(TARGETS)):
            p = np.mean([XGBClassifier(**XGB, random_state=s).fit(X[tr], y[tr, ti]).predict_proba(X[va])[:, 1]
                         for s in range(SEEDS)], 0)
            auc[ti, k] = roc_auc_score(y[va, ti], p)
    cache[key] = (auc, n_feat)
    print(f"[{time.time() - t0:5.0f}s] {name:38s} n_feat={n_feat:4d} "
          + " ".join(f"{t}={auc[i].mean():.3f}±{auc[i].std(ddof=1) / np.sqrt(len(folds)):.3f}" for i, t in enumerate(TARGETS)),
          flush=True)
    return cache[key]


def table(results, ref):
    rows = []
    for name, (auc, n_feat) in results.items():
        r = {"config": name, "n_feat": n_feat}
        for i, t in enumerate(TARGETS):
            d = auc[i] - results[ref][0][i]
            r[f"{t}_mean"] = auc[i].mean()
            r[f"{t}_diff_vs_ref"] = d.mean()
            r[f"{t}_diff_se"] = d.std(ddof=1) / np.sqrt(len(d))
            r[f"{t}_folds"] = np.round(auc[i], 3).tolist()
        rows.append(r)
    return pd.DataFrame(rows).set_index("config")


def best(tab, t):
    return tab[f"{t}_mean"].idxmax()


out = {"folds": [(int(tr[-1]) + 1, int(va[0]) + 1, int(va[-1]) + 1) for tr, va in folds], "xgb": XGB, "seeds": SEEDS}
print("walk-forward folds (train_end, val_start, val_end Time_step):", out["folds"])

# ---- stage 1: 표현 × 축소 (+ 기존 그룹별 맞춤 subset, full_raw = level/none)
s1 = {"tailored (기존 subset)": evaluate("tailored (기존 subset)", repr="tailored")}
for rep in REPRS:
    for red in ["none", "cluster", "cluster+pca"]:
        s1[f"{rep} / {red}"] = evaluate(f"{rep} / {red}", repr=rep, reduce=red)
tab1 = table(s1, "level / none")  # 기준: 원본 그대로(full_raw)
out["stage1"] = tab1.reset_index().to_dict("records")

# ---- stage 2: target별 stage1 최적 설정에서 그룹 제외/단독
stage2 = {}
for ti, t in enumerate(TARGETS):
    b = best(tab1, t)
    cfg = {"repr": "tailored"} if b.startswith("tailored") else dict(zip(["repr", "reduce"], b.split(" / ")))
    s2 = {f"[{b}] all": s1[b]}
    for g in list(groups) + ["CAL"]:
        inc = [x for x in groups if x != g]
        s2[f"[{b}] -{g}"] = evaluate(f"[{b}] -{g}", **cfg, include=inc if g != "CAL" else None, cal=(g != "CAL"))
    for g in groups:
        s2[f"[{b}] only {g}+CAL"] = evaluate(f"[{b}] only {g}+CAL", **cfg, include=[g])
    tab2 = table(s2, f"[{b}] all")
    stage2[t] = (cfg, tab2)
    out[f"stage2_{t}"] = tab2.reset_index().to_dict("records")

# ---- stage 3: target별 stage1 최적 설정 + EXAONE
for ti, t in enumerate(TARGETS):
    cfg, _ = stage2[t]
    b = best(tab1, t)
    s3 = {f"[{b}]": s1[b], f"[{b}] +exaone": evaluate(f"[{b}] +exaone", **cfg, extras=[exaone])}
    out[f"stage3_{t}"] = table(s3, f"[{b}]").reset_index().to_dict("records")

json.dump(out, open(ROOT / "results/ablation.json", "w"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 200)
cols = lambda t: [f"{t}_mean", f"{t}_diff_vs_ref", f"{t}_diff_se"]
print("\n== Stage 1 (기준: level / none = 원본 그대로)")
print(tab1[["n_feat", *cols("Target_1"), *cols("Target_2")]].round(3).to_string())
for t in TARGETS:
    print(f"\n== Stage 2 {t}")
    print(pd.DataFrame(out[f"stage2_{t}"]).set_index("config")[["n_feat", *cols(t)]].round(3).to_string())
    print(f"\n== Stage 3 {t}")
    print(pd.DataFrame(out[f"stage3_{t}"]).set_index("config")[["n_feat", *cols(t)]].round(3).to_string())
print(f"total {time.time() - t0:.0f}s")
