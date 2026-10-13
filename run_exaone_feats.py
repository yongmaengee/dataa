"""EXAONE Finance 1.0 zero-shot 예측을 causal rolling 방식으로 만들어 feature로 저장.

각 날짜 t에서 t까지의 값만 context로 쓰고(마지막 512개), horizon=3 분위수 예측을 생성.
feature: {col}_ex_d3 = 중앙값(h=3) - 현재값, {col}_ex_unc = q90 - q10 (h=3).
대상 계열: FI(금융, 모델의 학습 도메인) + SH1 + KE. MIN_CTX일 미만 구간은 NaN.

uv run python run_exaone_feats.py   →   results/exaone_feats.csv
"""
import os
import time

import numpy as np
import pandas as pd

from src.data import ROOT, load

for line in open(ROOT / ".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1)
        v = v.strip().strip("\"'")
        if v:
            os.environ.setdefault(k, v)

from exaone_forecast.finance import from_pretrained  # noqa: E402

MIN_CTX, H, DAYS_PER_BATCH = 30, 3, 16

df, _ = load()
cols = [c for c in df.columns if c.startswith(("FI_", "SH1_", "KE_"))]
cols = [c for c in cols if df[c].std() > 0]
A = df[cols].to_numpy(np.float64)  # (N, C)
N, C = A.shape

fc = from_pretrained(device="mps")
qs = list(fc.quantiles)
i50, i10, i90 = qs.index(0.5), qs.index(0.1), qs.index(0.9)

d3 = np.full((N, C), np.nan)
unc = np.full((N, C), np.nan)
days = np.arange(MIN_CTX - 1, N)
t0 = time.time()
for b in range(0, len(days), DAYS_PER_BATCH):
    chunk = days[b:b + DAYS_PER_BATCH]
    series = [A[max(0, t + 1 - fc.context_length):t + 1, c] for t in chunk for c in range(C)]  # t까지만 사용
    q = fc.predict(series, horizon=H)[:, :, H - 1].reshape(len(chunk), C, -1)
    d3[chunk] = q[:, :, i50] - A[chunk]
    unc[chunk] = q[:, :, i90] - q[:, :, i10]
    if b % (DAYS_PER_BATCH * 10) == 0:
        print(f"[{time.time() - t0:5.0f}s] day {chunk[-1] + 1}/{N}", flush=True)

out = pd.concat([pd.DataFrame(d3, columns=[f"{c}_ex_d3" for c in cols]),
                 pd.DataFrame(unc, columns=[f"{c}_ex_unc" for c in cols])], axis=1)
out.insert(0, "Time_step", df["Time_step"].to_numpy())
out.to_csv(ROOT / "results/exaone_feats.csv", index=False, encoding="utf-8")
print(f"saved results/exaone_feats.csv {out.shape} ({time.time() - t0:.0f}s)")
