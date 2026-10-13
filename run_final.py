"""target별 종합 비교 (walk-forward 5 fold, XGBoost 고정) → 최종 모델 선택 → 제출 파일.

후보 (target별 확정 표현 PRESET 기준):
  xgb[base]                       Stage 1 확정 표현만
  xgb[base+domain]                + 도메인 파생변수 (스프레드, 풍향 벡터, 일교차, 강수 여부, 미세먼지 7일 평균)
  xgb[base+domain+exaone]         + EXAONE 예측변화량·불확실성 + surprise
  xgb[base+domain+experts]        + TabPFN 그룹 전문가 OOF 확률 (stacking, results/tabpfn_experts.csv)
  xgb[base+domain+exaone+experts]
  tabpfn[base]                    TabPFN 'all' 전문가의 블록 예측 그대로 (단독 TabPFN 후보)

사전 실행: uv run python run_tabpfn_experts.py
uv run python run_final.py [--sub-idx 3]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.ablation import PRESET, exaone_surprise, make_features, walk_forward
from src.data import ROOT, TARGETS, group_columns, load

XGB = dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5,
           min_child_weight=5, reg_lambda=1.0, n_jobs=8)
SEEDS = 5

p = argparse.ArgumentParser()
p.add_argument("--sub-idx", type=int, default=3)
args = p.parse_args()

t0 = time.time()
df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, TARGETS].to_numpy().astype(int)
n = int(is_train.sum())
train_idx, test_idx = np.where(is_train)[0], np.where(~is_train)[0]
folds = walk_forward(n)

ex = pd.read_csv(ROOT / "results/exaone_feats.csv").drop(columns="Time_step")
exaone = [ex, exaone_surprise(df, ex)]
experts = pd.read_csv(ROOT / "results/tabpfn_experts.csv")
assert (experts["Time_step"].to_numpy() == df["Time_step"].to_numpy()).all()


def xgb_predict(X, tr, te, ti):
    return np.mean([XGBClassifier(**XGB, random_state=s).fit(X[tr], y[tr, ti]).predict_proba(X[te])[:, 1]
                    for s in range(SEEDS)], 0)


def candidates(t):
    exp_t = experts[[c for c in experts.columns if c.startswith(f"EXP_{t}_")]]
    return {
        "xgb[base]": dict(domain=False, extras=[]),
        "xgb[base+domain]": dict(domain=True, extras=[]),
        "xgb[base+domain+exaone]": dict(domain=True, extras=exaone),
        "xgb[base+domain+experts]": dict(domain=True, extras=[exp_t]),
        "xgb[base+domain+exaone+experts]": dict(domain=True, extras=[*exaone, exp_t]),
    }


results = {"xgb": XGB, "seeds": SEEDS, "preset": PRESET, "folds": [(int(va[0]) + 1, int(va[-1]) + 1) for _, va in folds],
           "tables": {}, "chosen": {}}
sub = {}
for ti, t in enumerate(TARGETS):
    cfg = PRESET[t]
    auc = {}
    for name, c in candidates(t).items():
        a = []
        for tr, va in folds:
            fit_mask = np.zeros(len(df), bool)
            fit_mask[tr] = True
            X = make_features(df, fit_mask, groups, **cfg, **c).to_numpy(np.float32)
            a.append(roc_auc_score(y[va, ti], xgb_predict(X, tr, va, ti)))
        auc[name] = np.array(a)
        print(f"[{time.time() - t0:5.0f}s] {t} {name:34s} n_feat={X.shape[1]:5d} "
              f"AUC={np.mean(a):.3f}±{np.std(a, ddof=1) / np.sqrt(len(a)):.3f} folds={np.round(a, 3).tolist()}", flush=True)
    col = experts[f"EXP_{t}_all"].to_numpy()
    auc["tabpfn[base]"] = np.array([roc_auc_score(y[va, ti], col[va]) for _, va in folds])
    print(f"[{time.time() - t0:5.0f}s] {t} {'tabpfn[base]':34s} AUC={auc['tabpfn[base]'].mean():.3f} "
          f"folds={np.round(auc['tabpfn[base]'], 3).tolist()}", flush=True)

    ref = auc["xgb[base]"]
    tab = pd.DataFrame({name: {"mean": a.mean(), "se": a.std(ddof=1) / np.sqrt(len(a)),
                               "diff_vs_base": (a - ref).mean(), "diff_se": (a - ref).std(ddof=1) / np.sqrt(len(a)),
                               "min_fold": a.min()} for name, a in auc.items()}).T
    results["tables"][t] = tab.reset_index().rename(columns={"index": "model"}).to_dict("records")
    best = tab["mean"].idxmax()
    results["chosen"][t] = best
    print(f"\n== {t} (기준: xgb[base])\n{tab.round(3).to_string()}\n→ 선택: {best}\n", flush=True)

    # 전체 train 재학습 → test 예측
    if best == "tabpfn[base]":
        sub[f"pred_{ti + 1}"] = col[test_idx]
    else:
        X = make_features(df, is_train, groups, **cfg, **candidates(t)[best]).to_numpy(np.float32)
        sub[f"pred_{ti + 1}"] = xgb_predict(X, train_idx, test_idx, ti)

sub = pd.DataFrame({"Time_step": df.loc[test_idx, "Time_step"].to_numpy(), **sub})
out = ROOT / "submissions"
for name, m, ref in [(f"public_3반_안중범_{args.sub_idx}.csv", 100, "sample_submission_public.csv"),
                     ("private_3반_안중범.csv", 200, "sample_submission_private.csv")]:
    part = sub.iloc[:m]
    assert len(part) == m and (part["Time_step"].to_numpy() == pd.read_csv(ROOT / ref)["Time_step"].to_numpy()).all()
    assert part[["pred_1", "pred_2"]].apply(lambda c: c.between(0, 1).all()).all() and part.notna().all().all()
    part.to_csv(out / name, index=False)
    print("saved", out / name)
json.dump(results, open(ROOT / "results/final.json", "w"), ensure_ascii=False, indent=1, default=float)
print(f"total {time.time() - t0:.0f}s")
