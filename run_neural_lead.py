"""T1 신경망 (lead feature 입력) walk-forward 5 fold. 전처리 동결, 기법만 변경.

입력 (src/lead.py):
  mlp/conv1d : Q = fi_al(FI 정렬 lead 90 + 요일 2) + FI 당일 표준화값 18 → 윈도우 W일
  xattn      : Q 위와 동일, KV = KE/SH/WE zscore 스트림 윈도우
규제·증강: pre-LN residual, dropout, weight decay, 가우시안 노이즈, feature dropout, 시점 dropout, early stopping(inner tail)
KD(--kd): loss = a*BCE(y) + (1-a)*BCE(p_teacher) (results/teacher_T1.csv, causal OOF). teacher 없는 행은 hard label만.
재현: seed 고정, torch.use_deterministic_algorithms 가능 범위(MPS는 완전 결정적이지 않음).

uv run python run_neural_lead.py [--models mlp conv1d xattn xattn_nocross] [--kd] [--seeds 5] [--quick]
→ results/neural_lead_T1{_kd}.json
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.ablation import walk_forward
from src.data import ROOT, load
from src.lead import MAX_LEAD, streams
from src.neural_lead import MODELS
from src.train import windows

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="*", default=["mlp", "conv1d", "xattn", "xattn_nocross"])
p.add_argument("--kd", action="store_true")
p.add_argument("--kd-alpha", type=float, default=0.3, help="hard label 가중치 (1-alpha가 teacher)")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--W", type=int, default=10)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
p.add_argument("--quick", action="store_true")
p.add_argument("--scheme", default="wf5", choices=["wf5", "wf10"])
args = p.parse_args()
CFG = dict(lr=1e-3, wd=2e-2, bs=64, max_epochs=120, patience=15, noise=0.15, feat_drop=0.2, step_drop=0.15, drop=0.3, d=64, L=2)
if args.quick:
    args.seeds, CFG["max_epochs"], CFG["patience"] = 1, 10, 4
dev = args.device

df, is_train = load()
y = df.loc[is_train, "Target_1"].to_numpy().astype(np.float32)
n = int(is_train.sum())
folds = walk_forward(n) if args.scheme == "wf5" else walk_forward(n, 10, 46)
teacher = None
if args.kd:
    tcsv = pd.read_csv(ROOT / "results/teacher_T1.csv")
    assert (tcsv["Time_step"].to_numpy() == df["Time_step"].to_numpy()).all()
    teacher = tcsv["p_teacher"].to_numpy(np.float32)[:n]


def prepare(fit_mask):
    S = streams(df, is_train, fit_mask)
    out, dims = {}, {}
    for g, A in S.items():
        A = A.fillna(0).to_numpy(np.float32)
        mu, sd = A[fit_mask].mean(0), A[fit_mask].std(0) + 1e-6
        out[g] = torch.as_tensor(windows(np.clip((A - mu) / sd, -5, 5), args.W), device=dev)
        dims[g] = A.shape[1]
    return out, dims


def augment(xb):
    out = {}
    for g, x in xb.items():
        x = x + CFG["noise"] * torch.randn_like(x)
        x = x * (torch.rand(x.shape[0], 1, x.shape[2], device=dev) > CFG["feat_drop"])
        m = (torch.rand(x.shape[0], x.shape[1], 1, device=dev) > CFG["step_drop"]).float()
        m[:, -1] = 1  # 당일 시점은 유지
        out[g] = x * m
    return out


def take(X, idx):
    return {g: x[idx] for g, x in X.items()}


@torch.no_grad()
def predict(model, X, idx):
    model.eval()
    return torch.sigmoid(model(take(X, torch.as_tensor(idx, device=dev)))).cpu().numpy()


def fit(name, X, dims, tr_in, es, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MODELS[name](dims, d=CFG["d"], L=CFG["L"], drop=CFG["drop"], W=args.W).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    yt = torch.as_tensor(y, device=dev)
    tt = torch.as_tensor(np.nan_to_num(teacher, nan=-1.0), device=dev) if teacher is not None else None
    best, best_state, wait = -1, None, 0
    for ep in range(CFG["max_epochs"]):
        model.train()
        for s in range(0, len(tr_in), CFG["bs"]):
            bi = torch.as_tensor(rng.permutation(tr_in)[s:s + CFG["bs"]], device=dev)
            logit = model(augment(take(X, bi)))
            loss = F.binary_cross_entropy_with_logits(logit, yt[bi])
            if tt is not None:
                has = tt[bi] >= 0
                if has.any():
                    soft = F.binary_cross_entropy_with_logits(logit[has], tt[bi][has])
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * soft
            opt.zero_grad(), loss.backward(), opt.step()
        auc = roc_auc_score(y[es], predict(model, X, es))
        if auc > best:
            best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        elif (wait := wait + 1) >= CFG["patience"]:
            break
    model.load_state_dict(best_state)
    return model


t0 = time.time()
tag = ("_kd" if args.kd else "") + ("" if args.scheme == "wf5" else f"_{args.scheme}")
res = {"config": {**CFG, **vars(args)}, "runs": {}}
for name in args.models:
    aucs, gates, oof, tr_aucs = [], [], np.full(n, np.nan), []
    for tr, va in folds:
        tr = tr[tr < va[0] - 3 - MAX_LEAD]  # 학습 행의 lead가 val 행을 참조하지 않도록
        fm = np.zeros(len(df), bool)
        fm[tr] = True
        X, dims = prepare(fm)
        tr_in, es = tr[:-(60 + MAX_LEAD)], tr[-60:]
        probs = []
        for seed in range(args.seeds):
            m = fit(name, X, dims, tr_in, es, seed)
            probs.append(predict(m, X, va))
            tr_aucs.append(roc_auc_score(y[tr_in], predict(m, X, tr_in)))
            if (g := m.gate() if hasattr(m, "gate") else None) is not None:
                gates.append(g)
        oof[va] = np.mean(probs, 0)
        aucs.append(roc_auc_score(y[va], oof[va]))
    r = dict(mean=float(np.mean(aucs)), se=float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))), folds=np.round(aucs, 3).tolist(),
             train_auc=float(np.mean(tr_aucs)), scheme=args.scheme,
             n_params=sum(p.numel() for p in m.parameters()), oof={"idx": np.where(~np.isnan(oof))[0].tolist(), "p": oof[~np.isnan(oof)].tolist()})
    if gates:
        r["gate_mean"] = float(np.mean(gates))
    res["runs"][name] = r
    print(f"[{time.time() - t0:5.0f}s] {name:14s} AUC={r['mean']:.3f}±{r['se']:.3f} train={r['train_auc']:.3f} folds={r['folds']} params={r['n_params']}"
          + (f" gate={r['gate_mean']:+.3f}" if gates else ""), flush=True)

json.dump(res, open(ROOT / f"results/neural_lead_T1{tag}.json", "w"), indent=1)
tab = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in res["runs"].items()}).T
print(f"\n== T1 lead 입력{' + KD' if args.kd else ''} (참고: TabPFN fi_k3+z 0.805±0.040, XGB fi_al 0.795±0.048)\n", tab.to_string())
print(f"total {time.time() - t0:.0f}s")
