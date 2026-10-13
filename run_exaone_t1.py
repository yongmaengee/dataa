"""T1 전용: EXAONE Finance 1.0을 직접 분류기로 사용 (zero-shot 상승 확률).

각 날짜 t에서 계열의 t까지 값(최근 512개)을 context로 horizon=3 분위수 예측 → P(x_{t+3} > x_t)를
분위수 격자 보간으로 계산. 대상 계열: FI 18개(금융, 모델 학습 도메인) + T1 label 계열(LABEL).
LABEL은 t 시점에 t-3까지만 알 수 있으므로 context를 t-3에서 끊고 3스텝 뒤를 사용(누수 방지).

평가 (walk-forward 5 fold val 구간, Time_step 465~924):
  zero-shot        : 계열별 P(rise) 그대로의 AUC (학습 없음)
  US3 mean         : Nasdaq/Dow/Sp500 P(rise) 평균
  LR stack         : FI 18개 P(rise) → 로지스틱 회귀 (fold train으로 fit)
→ results/exaone_t1.json, results/exaone_t1_prise.csv
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.ablation import walk_forward
from src.data import ROOT, group_columns, load
from src.tabpfn_util import load_env

load_env()
from exaone_forecast.finance import from_pretrained  # noqa: E402

MIN_CTX, H, DAYS_PER_BATCH = 30, 3, 32

df, is_train = load()
y = df.loc[is_train, "Target_1"].to_numpy().astype(int)
n = len(y)
folds = walk_forward(n)
fi_cols = group_columns(df)["FI"]
fc = from_pretrained(device="mps")
levels = np.asarray(fc.quantiles)


def p_rise(q, x_now):
    """q: (n, 21) horizon H 시점 분위수 값, x_now: (n,) 현재값 → P(x_{t+H} > x_now) = 1 - F(x_now)."""
    return np.array([1 - np.interp(x, np.sort(qi), levels[np.argsort(qi)], left=0.01, right=0.99) for qi, x in zip(q, x_now)])


t0 = time.time()
N = len(df)
P = pd.DataFrame({"Time_step": df["Time_step"].to_numpy()})

# (a) feature 계열: context = ..t, 예측 t+3
A = df[fi_cols].to_numpy(np.float64)
out = np.full((N, len(fi_cols)), np.nan)
for b in range(MIN_CTX - 1, N, DAYS_PER_BATCH):
    days = np.arange(b, min(b + DAYS_PER_BATCH, N))
    series = [A[max(0, t + 1 - fc.context_length):t + 1, c] for t in days for c in range(len(fi_cols))]
    q = fc.predict(series, horizon=H)[:, :, H - 1].reshape(len(days), len(fi_cols), -1)
    for j in range(len(fi_cols)):
        out[days, j] = p_rise(q[:, j], A[days, j])
    if b % (DAYS_PER_BATCH * 8) == MIN_CTX - 1:
        print(f"[{time.time() - t0:4.0f}s] FI day {days[-1] + 1}/{N}", flush=True)
for j, c in enumerate(fi_cols):
    P[f"{c}_prise"] = out[:, j]

# (b) label 계열: t 시점에 알 수 있는 건 y[..t-3] → context = y[:t-2], 3스텝 뒤 = t
yl = y.astype(np.float64)
lab = np.full(N, np.nan)
days = np.arange(MIN_CTX + H, n)
for b in range(0, len(days), DAYS_PER_BATCH * 4):
    d = days[b:b + DAYS_PER_BATCH * 4]
    series = [yl[max(0, t - H + 1 - fc.context_length):t - H + 1] for t in d]
    q = fc.predict(series, horizon=H)[:, :, H - 1]
    lab[d] = np.clip(q.mean(1), 0, 1)  # 0/1 계열의 분위수 평균 ≈ 기대값 ≈ P(up)
P["LABEL_prise"] = lab
P.to_csv(ROOT / "results/exaone_t1_prise.csv", index=False)
print(f"[{time.time() - t0:4.0f}s] P(rise) 생성 완료", flush=True)

# 평가
Ptr = P.iloc[:n]
res = {"zero_shot": {}, "stack": {}}
for c in [*[f"{c}_prise" for c in fi_cols], "LABEL_prise"]:
    a = [roc_auc_score(y[va], Ptr[c].to_numpy()[va]) for _, va in folds]
    res["zero_shot"][c] = dict(mean=float(np.mean(a)), folds=np.round(a, 3).tolist())
us3 = Ptr[[f"FI_{k}_prise" for k in ["Nasdaq", "Dow", "Sp500"]]].mean(1).to_numpy()
a = [roc_auc_score(y[va], us3[va]) for _, va in folds]
res["zero_shot"]["US3_mean"] = dict(mean=float(np.mean(a)), folds=np.round(a, 3).tolist())

Xs = Ptr[[f"{c}_prise" for c in fi_cols]].fillna(0.5).to_numpy()
a = []
for tr, va in folds:
    lr = LogisticRegression(C=0.1, max_iter=1000).fit(Xs[tr], y[tr])
    a.append(roc_auc_score(y[va], lr.predict_proba(Xs[va])[:, 1]))
res["stack"]["LR on 18 FI P(rise)"] = dict(mean=float(np.mean(a)), se=float(np.std(a, ddof=1) / np.sqrt(5)), folds=np.round(a, 3).tolist())

json.dump(res, open(ROOT / "results/exaone_t1.json", "w"), indent=1)
zs = pd.DataFrame(res["zero_shot"]).T.sort_values("mean", ascending=False)
print("\n== zero-shot P(rise) AUC (walk-forward val 구간)\n", zs.to_string())
print("\n== stack\n", pd.DataFrame(res["stack"]).T.to_string())
print(f"total {time.time() - t0:.0f}s")
