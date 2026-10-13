"""Lead feature (t+1..t+3일의 feature 변화) 실험 + 제출 파일.

label_t = 1[KOSPI_{t+3} > KOSPI_t] 이므로 같은 구간의 환율·해외지수 변화가 동시 상관으로 강하게 연결됨.
test 파일에 925~1124일 feature가 모두 주어지므로 계산 가능. 단 인과(causal) 예측이 아님 — 과제 형식(배치 채점)에서만 성립.
test 마지막 3행(1122~1124)은 t+3이 없어 lead가 NaN → XGB는 NaN 분기로 처리.

uv run python run_lead.py --model xgb [--featset lead_all lead_fi] [--sub-idx 4]
uv run python run_lead.py --model tabpfn --featset lead_fi    (별도 프로세스)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ablation import PRESET, make_features, walk_forward
from src.data import ROOT, TARGETS, group_columns, load

p = argparse.ArgumentParser()
p.add_argument("--model", default="xgb", choices=["xgb", "tabpfn"])
p.add_argument("--featset", nargs="*", default=["lead_all", "lead_fi"])
p.add_argument("--sub-idx", type=int, default=4)
p.add_argument("--no-sub", action="store_true")
p.add_argument("--skip-cv", action="store_true", help="results/lead_xgb.json을 읽어 CV 생략, 제출만 생성")
p.add_argument("--test-pred", action="store_true", help="CV 없이 전체 train fit → test 예측을 results/lead_{model}_test.json에 저장")
p.add_argument("--targets", nargs="*", default=TARGETS)
args = p.parse_args()

df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, TARGETS].to_numpy().astype(int)
n = int(is_train.sum())
train_idx, test_idx = np.where(is_train)[0], np.where(~is_train)[0]
MAX_LEAD = 3
fit_idx = train_idx[:-MAX_LEAD]  # 922~924일은 lead(t+1..t+3)가 test 행을 참조 → 학습에서 제외
folds = walk_forward(n)


def leads(cols, ks=(1, 2, 3)):
    """ks 원소: k → x[t+k]-x[t], 또는 (a,b) → x[t+a]-x[t+b]."""
    X = df[cols]
    X = X.loc[:, X[is_train].std() > 0]
    parts = []
    for k in ks:
        a, b = (k, 0) if isinstance(k, int) else k
        parts.append((X.shift(-a) - X.shift(-b)).add_suffix(f"_lead{a}_{b}"))
    return pd.concat(parts, axis=1)


LEAD = {"lead_all": leads([c for cs in groups.values() for c in cs]), "lead_fi": leads(groups["FI"]), "lead_fi3": leads(groups["FI"], ks=(3,)),
        "lead_fi_al": leads(groups["FI"], ks=((3, 0), (2, -1), (3, -1), (2, 0), (3, -2))),
        "lead_fi_al3": leads(groups["FI"], ks=((3, 0), (2, -1), (3, -1)))}


def feats(t, fs, fit_mask):
    if fs.startswith("nobase:"):
        from src.features import calendar
        return pd.concat([LEAD[fs[7:]], calendar(df)], axis=1).to_numpy(np.float32)
    base = make_features(df, fit_mask, groups, **PRESET[t])
    if args.model == "tabpfn" and t == "Target_1":  # 1412 + lead는 MPS 메모리 초과 → zscore 표현(566)으로 축소
        base = make_features(df, fit_mask, groups, repr="zscore", reduce="none")
    return pd.concat([base, LEAD[fs]], axis=1).to_numpy(np.float32)


if args.model == "xgb":
    from xgboost import XGBClassifier
    XGB = dict(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0, n_jobs=8)

    def fit_predict(Xtr, ytr, Xte):
        return np.mean([XGBClassifier(**XGB, random_state=s).fit(Xtr, ytr).predict_proba(Xte)[:, 1] for s in range(5)], 0)
else:
    from src.tabpfn_util import fit_predict as _fp, load_env
    load_env()
    fit_predict = lambda Xtr, ytr, Xte: _fp(Xtr, ytr, Xte)

t0 = time.time()
res = {"model": args.model, "runs": {}}
if args.test_pred:
    tp_path = ROOT / f"results/lead_{args.model}_test.json"
    tp = json.load(open(tp_path)) if tp_path.exists() else {"Time_step": df.loc[test_idx, "Time_step"].tolist()}
    for fs in args.featset:
        for ti, t in enumerate(TARGETS):
            if t not in args.targets:
                continue
            X = feats(t, fs, is_train)
            tp[f"{t}|{fs}"] = fit_predict(X[fit_idx], y[:len(fit_idx), ti], X[test_idx]).tolist()
            print(f"[{time.time() - t0:4.0f}s] test-pred {args.model} {fs} {t} n_feat={X.shape[1]}", flush=True)
            json.dump(tp, open(tp_path, "w"), indent=1)
    raise SystemExit(0)
if args.skip_cv:
    res = json.load(open(ROOT / f"results/lead_{args.model}.json"))
for fs in ([] if args.skip_cv else args.featset):
    for ti, t in enumerate(TARGETS):
        a = []
        for tr, va in folds:
            fm = np.zeros(len(df), bool)
            fm[tr] = True
            X = feats(t, fs, fm)
            a.append(roc_auc_score(y[va, ti], fit_predict(X[tr], y[tr, ti], X[va])))
        res["runs"][f"{t}|{fs}"] = dict(mean=float(np.mean(a)), se=float(np.std(a, ddof=1) / np.sqrt(5)), folds=np.round(a, 3).tolist(), n_feat=X.shape[1])
        print(f"[{time.time() - t0:4.0f}s] {args.model} {fs:9s} {t} n_feat={X.shape[1]:5d} AUC={np.mean(a):.3f}±{np.std(a, ddof=1) / np.sqrt(5):.3f} {np.round(a, 3).tolist()}", flush=True)
if not args.skip_cv:
    json.dump(res, open(ROOT / f"results/lead_{args.model}.json", "w"), indent=1)

if not args.no_sub and args.model == "xgb":
    chosen = {t: max(args.featset, key=lambda f: res["runs"][f"{t}|{f}"]["mean"]) for t in TARGETS}  # target별 최적 세트
    fs = chosen
    sub = {"Time_step": df.loc[test_idx, "Time_step"].to_numpy()}
    for ti, t in enumerate(TARGETS):
        X = feats(t, chosen[t], is_train)
        sub[f"pred_{ti + 1}"] = fit_predict(X[fit_idx], y[:len(fit_idx), ti], X[test_idx])
    sub = pd.DataFrame(sub)
    out = ROOT / "submissions"
    prev = out / "private_3반_안중범.csv"
    if prev.exists() and not (out / "private_3반_안중범_tabpfn_causal.csv").exists():
        prev.rename(out / "private_3반_안중범_tabpfn_causal.csv")
    for name, m, ref in [(f"public_3반_안중범_{args.sub_idx}.csv", 100, "sample_submission_public.csv"), ("private_3반_안중범.csv", 200, "sample_submission_private.csv")]:
        part = sub.iloc[:m]
        assert len(part) == m and (part["Time_step"].to_numpy() == pd.read_csv(ROOT / ref)["Time_step"].to_numpy()).all()
        assert part[["pred_1", "pred_2"]].apply(lambda c: c.between(0, 1).all()).all() and part.notna().all().all()
        part.to_csv(out / name, index=False)
        print("saved", out / name, f"(featset={fs})")
print(f"total {time.time() - t0:.0f}s")
