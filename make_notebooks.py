"""재현 노트북 2개 생성: final_nokd.ipynb (외부 모델 없음) / final_kd.ipynb (TabPFN teacher KD).
uv run python make_notebooks.py --t1-model xattn [--seeds 5]
노트북은 train.csv + test_private.csv만으로 private 제출 파일을 처음부터 다시 만든다 (시드 고정, 버전 출력).
"""
import argparse

import nbformat as nbf

p = argparse.ArgumentParser()
p.add_argument("--t1-model", default="conv1d")
p.add_argument("--no-blend", action="store_true", help="T1에 XGB fi_al 순위평균 블렌드를 하지 않음")
p.add_argument("--seeds", type=int, default=5)
a = p.parse_args()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def cells(kd):
    tag = "kd" if kd else "nokd"
    c = [md(f"""# Private Score 재현 노트북 ({'KD: TabPFN teacher 증류' if kd else '외부 모델 없음'})

- 입력: `train.csv`, `test_private.csv`만 사용 (외부 데이터 없음)
- 평가 데이터는 **학습에 사용하지 않음**: 학습 행은 train 1~921 (922~924행은 lead feature가 test 행을 참조하므로 제외)
- Target_1: `{a.t1_model}` (lead feature 입력 신경망, 시드 {a.seeds}개 평균){' + TabPFN causal-OOF teacher 증류' if kd else ''}{'' if a.no_blend else ' + XGBoost(fi_al) 순위 평균 블렌드'}
- 검증(walk-forward): 블렌드 wf5 0.804 / wf10 0.806, fold 최저 0.695 / 0.649 (`results/neural_lead_T1*.json`, `results/lead_val_xgb.json`)
- Target_2: XGBoost (level+change + 전체 lead k=1..3), 시드 5개 평균
- 검증 근거: walk-forward 5 fold (`record.md`, `report.html`)
- 재현: 시드 고정. MPS는 완전 결정적이지 않아 소수점 셋째 자리 수준의 차이가 있을 수 있음. 완전 재현이 필요하면 `DEVICE="cpu"`
"""),
          code(f"""import copy, json, random, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, xgboost, sklearn
from xgboost import XGBClassifier
from src.data import ROOT, TARGETS, load
from src.lead import MAX_LEAD, lead_frame, streams
from src.neural_lead import MODELS
from src.train import windows

SEEDS, W, DEVICE = {a.seeds}, 10, "mps" if torch.backends.mps.is_available() else "cpu"
T1_MODEL, KD, KD_ALPHA, BLEND = "{a.t1_model}", {kd}, 0.3, {not a.no_blend}
CFG = dict(lr=1e-3, wd=2e-2, bs=64, max_epochs=120, patience=15, noise=0.15, feat_drop=0.2, step_drop=0.15, drop=0.3, d=64, L=2)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
print("python", sys.version.split()[0], "| torch", torch.__version__, "| xgboost", xgboost.__version__, "| sklearn", sklearn.__version__, "| device", DEVICE)

df, is_train = load()
y = df.loc[is_train, TARGETS].to_numpy().astype(np.float32)
n = int(is_train.sum())
train_idx, test_idx = np.where(is_train)[0], np.where(~is_train)[0]
fit_idx = train_idx[:-MAX_LEAD]          # 922~924 제외
fit_mask = np.zeros(len(df), bool); fit_mask[fit_idx] = True
print("train rows used:", len(fit_idx), "| test rows:", len(test_idx))"""),
          md("## Target_2: XGBoost"),
          code("""X2 = lead_frame(df, is_train, "all_k123+lc", fit_mask).to_numpy(np.float32)
XGB = dict(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0, n_jobs=8)
pred_2 = np.mean([XGBClassifier(**XGB, random_state=s).fit(X2[fit_idx], y[:len(fit_idx), 1]).predict_proba(X2[test_idx])[:, 1] for s in range(5)], 0)
print("T2 n_feat", X2.shape[1], "| pred mean", pred_2.mean().round(3))"""),
          ]
    if kd:
        c += [md("## Teacher (TabPFN causal OOF) — `results/teacher_T1.csv`가 없으면 생성 (Prior Labs 토큰 필요)"),
              code("""tpath = ROOT / "results/teacher_T1.csv"
if not tpath.exists():
    import subprocess; subprocess.run([sys.executable, "run_teacher_oof.py"], check=True)
teacher = pd.read_csv(tpath)
assert (teacher["Time_step"].to_numpy() == df["Time_step"].to_numpy()).all()
teacher = teacher["p_teacher"].to_numpy(np.float32)[:n]
print("teacher rows with prob:", int(np.isfinite(teacher).sum()))""")]
    c += [md("## Target_1: lead feature 신경망"),
          code("""S = streams(df, is_train, fit_mask)
X, dims = {}, {}
for g, A in S.items():
    A = A.fillna(0).to_numpy(np.float32); mu, sd = A[fit_mask].mean(0), A[fit_mask].std(0) + 1e-6
    X[g] = torch.as_tensor(windows(np.clip((A - mu) / sd, -5, 5), W), device=DEVICE); dims[g] = A.shape[1]
take = lambda idx: {g: x[idx] for g, x in X.items()}
def augment(xb):
    out = {}
    for g, x in xb.items():
        x = x + CFG["noise"] * torch.randn_like(x)
        x = x * (torch.rand(x.shape[0], 1, x.shape[2], device=DEVICE) > CFG["feat_drop"])
        m = (torch.rand(x.shape[0], x.shape[1], 1, device=DEVICE) > CFG["step_drop"]).float(); m[:, -1] = 1
        out[g] = x * m
    return out
@torch.no_grad()
def predict(model, idx):
    model.eval(); return torch.sigmoid(model(take(torch.as_tensor(idx, device=DEVICE)))).cpu().numpy()
yt = torch.as_tensor(y[:, 0], device=DEVICE)
tt = torch.as_tensor(np.nan_to_num(teacher, nan=-1.0), device=DEVICE) if KD else None
tr_in, es = fit_idx[:-(60 + MAX_LEAD)], fit_idx[-60:]   # early stopping용 inner tail
def fit(seed):
    seed_all(seed); rng = np.random.default_rng(seed)
    model = MODELS[T1_MODEL](dims, d=CFG["d"], L=CFG["L"], drop=CFG["drop"], W=W).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    best, state, wait = -1, None, 0
    for ep in range(CFG["max_epochs"]):
        model.train()
        for s in range(0, len(tr_in), CFG["bs"]):
            bi = torch.as_tensor(rng.permutation(tr_in)[s:s + CFG["bs"]], device=DEVICE)
            logit = model(augment(take(bi))); loss = F.binary_cross_entropy_with_logits(logit, yt[bi])
            if tt is not None and (has := tt[bi] >= 0).any():
                loss = KD_ALPHA * loss + (1 - KD_ALPHA) * F.binary_cross_entropy_with_logits(logit[has], tt[bi][has])
            opt.zero_grad(); loss.backward(); opt.step()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[es, 0], predict(model, es))
        if auc > best: best, state, wait = auc, copy.deepcopy(model.state_dict()), 0
        elif (wait := wait + 1) >= CFG["patience"]: break
    model.load_state_dict(state); return model
t0 = time.time()
pred_1_nn = np.mean([predict(fit(s), test_idx) for s in range(SEEDS)], 0)
print(f"T1 {T1_MODEL} dims={dims} | pred mean {pred_1_nn.mean():.3f} | {time.time() - t0:.0f}s")"""),
          md("## Target_1: XGBoost(fi_al) + 순위 평균 블렌드"),
          code("""from scipy.stats import rankdata
X1 = lead_frame(df, is_train, "fi_al", fit_mask).fillna(0).to_numpy(np.float32)
pred_1_xgb = np.mean([XGBClassifier(**XGB, random_state=s).fit(X1[fit_idx], y[:len(fit_idx), 0]).predict_proba(X1[test_idx])[:, 1] for s in range(5)], 0)
# 순위 평균을 [0,1]로 정규화 (AUC는 순위만 사용)
pred_1 = ((rankdata(pred_1_nn) + rankdata(pred_1_xgb)) / 2 - 1) / (len(test_idx) - 1) if BLEND else pred_1_nn
print("T1 xgb n_feat", X1.shape[1], "| corr(nn, xgb) = %.3f" % np.corrcoef(pred_1_nn, pred_1_xgb)[0, 1], "| blend:", BLEND)"""),
          md("## 제출 파일"),
          code(f"""sub = pd.DataFrame({{"Time_step": df.loc[test_idx, "Time_step"].to_numpy(), "pred_1": pred_1, "pred_2": pred_2}})
out = ROOT / "submissions"; out.mkdir(exist_ok=True)
for name, m, ref in [("public_3반_안중범_{tag}.csv", 100, "sample_submission_public.csv"), ("private_3반_안중범_{tag}.csv", 200, "sample_submission_private.csv")]:
    part = sub.iloc[:m]
    assert len(part) == m and (part["Time_step"].to_numpy() == pd.read_csv(ROOT / ref)["Time_step"].to_numpy()).all()
    assert part[["pred_1", "pred_2"]].apply(lambda c: c.between(0, 1).all()).all() and part.notna().all().all()
    part.to_csv(out / name, index=False); print("saved", out / name)""")]
    return c


for kd in (False, True):
    nb = nbf.v4.new_notebook(cells=cells(kd), metadata={"kernelspec": {"name": "dataa", "display_name": "Python (dataa)", "language": "python"}})
    nbf.write(nb, f"final_{'kd' if kd else 'nokd'}.ipynb")
print("saved final_nokd.ipynb, final_kd.ipynb")
