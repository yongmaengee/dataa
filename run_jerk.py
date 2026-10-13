"""T1 기준 시간축 미분(1차=속도, 2차=가속도, 3차=jerk) 진단 → 1D conv(시간축 커널)가 의미 있는지.

(1) 구조 진단: 각 미분 차수의 lag-1 자기상관을 feature별로 구해 중앙값을 봄.
    원 계열이 random walk(d1이 백색)이면 과차분 서명이 나타남: d2 ≈ -0.50, d3 ≈ -0.67.
    관측값이 이 서명과 같으면 고차 미분은 새 정보 없는 노이즈 → 시간축 conv 커널이 d1 이상을 뽑아도 이득 없음.
(2) 예측 진단 (XGB 고정, walk-forward 5 fold): d1 / d1+d2 / d1+d2+d3 / d1 lag 0..3 (= 4차 이하 미분과 같은 정보)
→ results/jerk_T1.json
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.ablation import walk_forward
from src.data import ROOT, group_columns, load

XGB = dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5,
           min_child_weight=5, reg_lambda=1.0, n_jobs=8)
SEEDS = 3

df, is_train = load()
cols = [c for cs in group_columns(df).values() for c in cs]
raw = df[cols]
raw = raw.loc[:, raw[is_train].std() > 0]
y = df.loc[is_train, "Target_1"].to_numpy().astype(int)
folds = walk_forward(int(is_train.sum()))

d = {1: raw.diff()}
d[2] = d[1].diff()
d[3] = d[2].diff()

# (1) 구조 진단
struct = {}
for k, D in d.items():
    T = D[is_train]
    ac = T.apply(lambda s: s.autocorr(1))
    struct[f"d{k}"] = dict(lag1_autocorr_median=float(ac.median()), lag1_autocorr_q25=float(ac.quantile(.25)),
                           lag1_autocorr_q75=float(ac.quantile(.75)),
                           var_ratio_vs_d1_median=float((T.var() / d[1][is_train].var()).median()))
ref = {"random walk 과차분 서명": {"d1": 0.0, "d2": -0.5, "d3": -0.667}}
print("== 미분 차수별 lag-1 자기상관 (feature 중앙값 [q25, q75])")
for k in d:
    s = struct[f"d{k}"]
    print(f"  d{k}: {s['lag1_autocorr_median']:+.3f} [{s['lag1_autocorr_q25']:+.3f}, {s['lag1_autocorr_q75']:+.3f}]"
          f"   (random-walk 서명 {ref['random walk 과차분 서명'][f'd{k}']:+.3f})   var/var(d1)={s['var_ratio_vs_d1_median']:.2f}")
# 단일 feature 수준 신호: 각 미분 차수에서 |AUC-0.5| 상위 5개 평균 (train 전체, 참고용)
sig = {}
for k, D in d.items():
    T = D[is_train].fillna(0)
    a = np.array([abs(roc_auc_score(y, T[c]) - 0.5) for c in T.columns])
    sig[f"d{k}"] = float(np.sort(a)[-5:].mean())
print("  단일 feature |AUC-0.5| 상위5 평균:", {k: round(v, 3) for k, v in sig.items()})

# (2) 예측 진단
sets = {
    "d1": [d[1]],
    "d1+d2": [d[1], d[2]],
    "d1+d2+d3": [d[1], d[2], d[3]],
    "d1 lag0..3": [d[1].shift(i).add_suffix(f"_l{i}") for i in range(4)],
}
t0 = time.time()
pred = {}
for name, parts in sets.items():
    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    a = []
    for tr, va in folds:
        p = np.mean([XGBClassifier(**XGB, random_state=s).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1] for s in range(SEEDS)], 0)
        a.append(roc_auc_score(y[va], p))
    pred[name] = dict(mean=float(np.mean(a)), se=float(np.std(a, ddof=1) / np.sqrt(len(a))), folds=np.round(a, 3).tolist(), n_feat=X.shape[1])
    print(f"[{time.time() - t0:4.0f}s] {name:12s} n_feat={X.shape[1]:4d} AUC={pred[name]['mean']:.3f}±{pred[name]['se']:.3f} {pred[name]['folds']}", flush=True)

json.dump({"structure": struct, "single_feature_signal": sig, "xgb_walkforward": pred, "reference": ref},
          open(ROOT / "results/jerk_T1.json", "w"), ensure_ascii=False, indent=1)
print("saved results/jerk_T1.json")
