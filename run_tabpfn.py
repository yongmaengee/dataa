"""TabPFN 베이스라인 (별도 프로세스).

XGBoost와 같은 프로세스에서 돌리면 macOS에서 libomp 사본이 여러 개 로드되어
TabPFN(torch) OpenMP join에서 데드락이 발생하므로 노트북과 분리해서 실행.

uv run python run_tabpfn.py [--version v3.5|v3.5-fast] [--device mps|cpu] [--featsets subset full_raw ...]
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import ROOT, SCHEMES, TARGETS, group_columns, load, make_split
from src.features import featsets

for line in open(ROOT / ".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1)
        v = v.strip().strip("\"'")  # 따옴표 유무 모두 허용
        if v:
            os.environ.setdefault(k, v)

from tabpfn import TabPFNClassifier  # noqa: E402  (.env 로드 후 import)
from tabpfn.constants import ModelVersion  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--version", default="v3.5", choices=["v3.5", "v3.5-fast"])
p.add_argument("--device", default="mps")
p.add_argument("--featsets", nargs="*", default=None, help="기본: 사용 가능한 전체 세트")
args = p.parse_args()
version = ModelVersion(args.version)


def model(**kw):
    return TabPFNClassifier.create_default_for_version(version, **{"device": args.device, "random_state": 0,
                                                                   "ignore_pretraining_limits": True, **kw})


def fit_predict(Xtr, ytr, Xte):
    """MPS 메모리 부족 시 memory_saving_mode → CPU 순으로 재시도."""
    for kw in ({}, {"memory_saving_mode": True}, {"device": "cpu"}):
        try:
            return model(**kw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        except Exception as e:  # TabPFNMPSOutOfMemoryError 등
            if "out of memory" not in str(e).lower():
                raise
            print(f"  OOM → retry with {kw or 'memory_saving_mode'}", flush=True)
    raise RuntimeError("TabPFN OOM on all fallbacks")


t0 = time.time()
df, is_train = load()
sets = featsets(df, is_train, group_columns(df))
y = df.loc[is_train, TARGETS].to_numpy().astype(int)
res = {"config": vars(args), "runs": [], "val_preds": {}, "test_preds": {},
       "test_time_step": df.loc[~is_train, "Time_step"].tolist()}

for fs in args.featsets or list(sets):
    X_all = sets[fs].to_numpy(np.float32)
    X, X_test = X_all[is_train], X_all[~is_train]
    for scheme in SCHEMES:
        tr, va = make_split(y, scheme, seed=0)
        for ti, t in enumerate(TARGETS):
            pr = fit_predict(X[tr], y[tr, ti], X[va])
            auc = roc_auc_score(y[va, ti], pr)
            res["runs"].append(dict(scheme=scheme, target=t, variant="tabpfn", featset=fs, auc=float(auc)))
            res["val_preds"][f"{scheme}|{t}|tabpfn[{fs}]"] = {"idx": va.tolist(), "y": y[va, ti].tolist(), "p": pr.tolist()}
            print(f"[{time.time() - t0:5.0f}s] {fs:16s} {scheme:10s} {t} n_feat={X.shape[1]} auc={auc:.3f}", flush=True)
    # 참고용: 전체 train으로 fit한 test(925~1124) 예측 (제출 파일은 만들지 않음)
    res["test_preds"][fs] = {t: fit_predict(X, y[:, ti], X_test).tolist() for ti, t in enumerate(TARGETS)}

json.dump(res, open(ROOT / "results/tabpfn_results.json", "w"), indent=1)
print(f"saved results/tabpfn_results.json ({time.time() - t0:.0f}s)")
