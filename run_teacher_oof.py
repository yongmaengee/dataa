"""KD teacher: TabPFN(fi_k3+z) causal OOF 확률. 92일 블록마다 블록 시작-3일 이전 행으로만 fit. test는 train(마지막 3행 제외)으로 fit.
→ results/teacher_T1.csv (Time_step, p_teacher). KD 전용(외부 모델 의존); 비-KD 파이프라인은 사용하지 않음.
"""
import time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.ablation import walk_forward
from src.data import ROOT, load
from src.lead import MAX_LEAD, lead_frame
from src.tabpfn_util import fit_predict, load_env
load_env()
df, is_train = load(); y = df.loc[is_train, "Target_1"].to_numpy().astype(int); n = int(is_train.sum())
train_idx, test_idx = np.where(is_train)[0], np.where(~is_train)[0]
p = np.full(len(df), np.nan); t0 = time.time()
for tr, va in walk_forward(n, 8, 92):
    fm = np.zeros(len(df), bool); fm[tr] = True
    X = lead_frame(df, is_train, "fi_k3+z", fm).to_numpy(np.float32)
    p[va] = fit_predict(X[tr], y[tr], X[va]); print(f"[{time.time()-t0:4.0f}s] block {va[0]+1}-{va[-1]+1} auc={roc_auc_score(y[va], p[va]):.3f}", flush=True)
fit_idx = train_idx[:-MAX_LEAD]
X = lead_frame(df, is_train, "fi_k3+z", is_train).to_numpy(np.float32)
p[test_idx] = fit_predict(X[fit_idx], y[:len(fit_idx)], X[test_idx])
pd.DataFrame({"Time_step": df["Time_step"], "p_teacher": p}).to_csv(ROOT / "results/teacher_T1.csv", index=False)
print(f"saved results/teacher_T1.csv ({time.time()-t0:.0f}s)")
