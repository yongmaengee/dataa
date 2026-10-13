"""lead feature 후보 리스트 × 검증 방식 다양화.

후보: base(none/zscore/change+zscore) × lead 범위(FI/FX/ALL) × k(1..3 / 3 / 1..4)
검증: wf5(92일×5) / wf10(46일×10) / time20(마지막 20%) / random(참고, 누수) + wf5 OOF로 100일 슬라이딩 AUC 분포
uv run python run_lead_val.py --model xgb
uv run python run_lead_val.py --model tabpfn --cands C1 C2 ...   (별도 프로세스)
→ results/lead_val_{model}.json (OOF 포함)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit

from src.ablation import make_features, walk_forward
from src.data import ROOT, TARGETS, group_columns, load

p = argparse.ArgumentParser()
p.add_argument("--model", default="xgb", choices=["xgb", "tabpfn"])
p.add_argument("--cands", nargs="*", default=None)
p.add_argument("--targets", nargs="*", default=TARGETS)
p.add_argument("--schemes", nargs="*", default=["wf5", "wf10", "time20", "random"])
p.add_argument("--seeds", type=int, default=3)
args = p.parse_args()

df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, TARGETS].to_numpy().astype(int)
n = int(is_train.sum())
FX = [c for c in groups["FI"] if "FX" in c]
LEADSRC = {"FI": groups["FI"], "FX": FX, "ALL": [c for cs in groups.values() for c in cs]}

CANDS = {  # 이름: (base 표현, lead 소스, k 목록)
    "fi_k123+cz": ("change+zscore", "FI", (1, 2, 3)),   # 제출 _4/_5 XGB 구성
    "fi_k123+z": ("zscore", "FI", (1, 2, 3)),            # 제출 _5 TabPFN 구성
    "fi_k123": (None, "FI", (1, 2, 3)),
    "fi_k3+z": ("zscore", "FI", (3,)),
    "fi_k1234+z": ("zscore", "FI", (1, 2, 3, 4)),
    "fx_k123+z": ("zscore", "FX", (1, 2, 3)),
    "fx_k123": (None, "FX", (1, 2, 3)),
    "all_k123+cz": ("change+zscore", "ALL", (1, 2, 3)),
    "all_k123+lc": ("level+change", "ALL", (1, 2, 3)),  # T2 제출 구성
    "fi_al+z": ("zscore", "FI", ((3, 0), (2, -1), (3, -1), (2, 0), (3, -2))),  # 정렬 anchor 5종
    "fi_al": (None, "FI", ((3, 0), (2, -1), (3, -1), (2, 0), (3, -2))),
    "fi_al3+z": ("zscore", "FI", ((3, 0), (2, -1), (3, -1))),
}
cands = args.cands or list(CANDS)


def leads(cols, ks):
    """ks 원소: k → x[t+k]-x[t], 또는 (a,b) → x[t+a]-x[t+b] (정렬 anchor)."""
    X = df[cols]
    X = X.loc[:, X[is_train].std() > 0]
    parts = []
    for k in ks:
        a, b = (k, 0) if isinstance(k, int) else k
        parts.append((X.shift(-a) - X.shift(-b)).add_suffix(f"_lead{a}_{b}"))
    return pd.concat(parts, axis=1)


def feats(name, fit_mask):
    base, src, ks = CANDS[name]
    parts = [leads(LEADSRC[src], ks)]
    if base:
        parts.insert(0, make_features(df, fit_mask, groups, repr=base, reduce="none"))
    else:
        from src.features import calendar
        parts.append(calendar(df))
    return pd.concat(parts, axis=1).to_numpy(np.float32)


def splits(scheme):
    if scheme == "wf5":
        return walk_forward(n, 5, 92)
    if scheme == "wf10":
        return walk_forward(n, 10, 46)
    if scheme == "time20":
        cut = int(round(n * 0.8))
        return [(np.arange(cut - 3), np.arange(cut, n))]
    if scheme == "random":
        return [(np.sort(a), np.sort(b)) for a, b in ShuffleSplit(3, test_size=0.2, random_state=0).split(np.arange(n))]
    raise ValueError(scheme)


if args.model == "xgb":
    from xgboost import XGBClassifier
    XGB = dict(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0, n_jobs=8)
    fit_predict = lambda Xtr, ytr, Xte: np.mean([XGBClassifier(**XGB, random_state=s).fit(Xtr, ytr).predict_proba(Xte)[:, 1] for s in range(args.seeds)], 0)
else:
    from src.tabpfn_util import fit_predict as _fp, load_env
    load_env()
    fit_predict = lambda Xtr, ytr, Xte: _fp(Xtr, ytr, Xte)


def sliding(oof, yy, w=100, step=10):
    a = [roc_auc_score(yy[s:s + w], oof[s:s + w]) for s in range(0, len(yy) - w + 1, step) if len(set(yy[s:s + w])) > 1]
    return dict(min=float(np.min(a)), p25=float(np.percentile(a, 25)), median=float(np.median(a)), max=float(np.max(a)), n=len(a))


t0 = time.time()
out_path = ROOT / f"results/lead_val_{args.model}.json"
res = json.load(open(out_path)) if out_path.exists() else {}
for name in cands:
    for ti, t in enumerate(TARGETS):
        if t not in args.targets:
            continue
        r = res.setdefault(f"{t}|{name}", {})
        for sc in args.schemes:
            if sc in r:
                continue
            aucs, oof = [], np.full(n, np.nan)
            for tr, va in splits(sc):
                fm = np.zeros(len(df), bool)
                fm[tr] = True
                X = feats(name, fm)
                pr = fit_predict(X[tr], y[tr, ti], X[va])
                aucs.append(roc_auc_score(y[va, ti], pr))
                if sc == "wf5":
                    oof[va] = pr
            r[sc] = dict(mean=float(np.mean(aucs)), se=float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else None, folds=np.round(aucs, 3).tolist())
            if sc == "wf5":
                m = ~np.isnan(oof)
                r["oof_wf5"] = {"idx": np.where(m)[0].tolist(), "p": oof[m].tolist()}
                r["sliding100"] = sliding(oof[m], y[m, ti])
            r["n_feat"] = int(X.shape[1])
            json.dump(res, open(out_path, "w"), indent=1)
            print(f"[{time.time() - t0:4.0f}s] {args.model} {t} {name:14s} {sc:7s} n_feat={X.shape[1]:5d} AUC={np.mean(aucs):.3f} {np.round(aucs, 3).tolist()}", flush=True)
print(f"total {time.time() - t0:.0f}s")
