"""Group-wise cross-attention 실험 단일 진입점.

uv run python run_xattn.py            # 전체 실행
uv run python run_xattn.py --quick    # 1 seed, 짧은 학습으로 동작 확인
uv run python run_xattn.py --full     # 그룹 분리 없이 원본 feature 전체를 단일 스트림으로 (q_only, concat_mlp만)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.data import ROOT, SCHEMES, TARGETS, group_columns, inner_split, load, make_split
from src.features import build
from src.train import diagnostics, fit, predict, prepare
from src.xattn import ConcatMLP, XAttnNet

Q_BASE = {"Target_1": "FI", "Target_2": "SH"}  # target별 query 그룹
MERGE_CCA = 0.7  # Q 그룹과 diff7-CCA가 이 이상이면 중복 정보로 보고 KV가 아닌 Q 스트림에 합침

p = argparse.ArgumentParser()
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--W", type=int, default=20)
p.add_argument("--tta", type=int, default=16)
p.add_argument("--sub-idx", type=int, default=2, help="public 제출 회차 번호")
p.add_argument("--quick", action="store_true")
p.add_argument("--full", action="store_true", help="원본 feature 전체를 단일 Q 스트림으로 (cross-attn 없음)")
args = p.parse_args()

CFG = dict(d=32, heads=4, dropout=0.2, ffn_C=2000, lr=1e-3, wd=1e-2, bs=64, max_epochs=150, patience=20,
           noise=0.1, feat_drop=0.1, tok_drop=0.1)
if args.quick:
    args.seeds, CFG["max_epochs"], CFG["patience"] = 1, 15, 5
VARIANTS = {
    "xattn": lambda dims, kv: XAttnNet(dims, kv, args.W, CFG["d"], CFG["heads"], CFG["dropout"], CFG["ffn_C"]),
    "q_only": lambda dims, kv: XAttnNet(dims, kv, args.W, CFG["d"], CFG["heads"], CFG["dropout"], CFG["ffn_C"], use_cross=False),
    "concat_mlp": lambda dims, kv: ConcatMLP(dims, kv, CFG["dropout"]),
}

if args.full:  # KV가 없으므로 cross-attn 변형은 제외
    VARIANTS = {k: v for k, v in VARIANTS.items() if k != "xattn"}

t0 = time.time()
df, is_train = load()
feats, cal, report = build(df, is_train, group_columns(df))
(ROOT / "results").mkdir(exist_ok=True)
json.dump(report, open(ROOT / "results/collinearity.json", "w"), ensure_ascii=False, indent=1)
cca = pd.DataFrame(report["cca_diff7"])
print("== 공선성 축소:", {g: (r["n_raw"], r["n_cluster"], r.get("n_pca")) for g, r in report["within"].items()},
      "(raw, cluster, pca)")
print("== 그룹 간 CCA (7일 차분)\n", cca.round(2))

y_full = df[TARGETS].to_numpy(dtype=np.float32)  # test 행은 NaN, 학습/평가에서 사용하지 않음
y = y_full[is_train].astype(int)
test_idx = np.where(~is_train)[0]

# target별 Q/KV 배정 + 스트림 구성 (모든 스트림에 calendar를 붙여 시점 정렬 단서 제공)
setup = {}
raw = df[[c for cols in group_columns(df).values() for c in cols]]
for t in ([] if not args.full else TARGETS):
    streams = {"Q": pd.concat([raw, cal], axis=1).to_numpy()}
    setup[t] = dict(q=["ALL_RAW"], kv=[], streams=streams, dims={"Q": streams["Q"].shape[1]})
    print(f"== {t}: Q=ALL_RAW (단일 스트림) dims={setup[t]['dims']}")
for t in ([] if args.full else TARGETS):
    q0 = Q_BASE[t]
    others = [g for g in feats if g != q0]
    q = [q0] + [g for g in others if cca.loc[q0, g] >= MERGE_CCA]
    kv = [g for g in others if g not in q]
    streams = {"Q": pd.concat([*(feats[g] for g in q), cal], axis=1).to_numpy()}
    streams.update({g: pd.concat([feats[g], cal], axis=1).to_numpy() for g in kv})
    setup[t] = dict(q=q, kv=kv, streams=streams, dims={g: a.shape[1] for g, a in streams.items()})
    print(f"== {t}: Q={q} KV={kv} dims={setup[t]['dims']}")

results = {"config": {**CFG, **vars(args)}, "assignment": {t: {"Q": s["q"], "KV": s["kv"]} for t, s in setup.items()},
           "runs": [], "val_preds": {}}
best_epochs = {t: [] for t in TARGETS}

for scheme in SCHEMES:
    tr, va = make_split(y, scheme, seed=0)
    tr_in, es = inner_split(y, tr, scheme)
    for ti, t in enumerate(TARGETS):
        s = setup[t]
        X = prepare(s["streams"], tr, args.W, args.device)
        yt = y_full[:, ti]
        for vname, mk in VARIANTS.items():
            P = {"plain": [], "tta": [], "kv_shuffle": []}
            per_seed, diags, eps = [], [], []
            for seed in range(args.seeds):
                model, ep = fit(lambda: mk(s["dims"], s["kv"]), X, yt, tr_in, es, CFG, seed, args.device)
                P["plain"].append(predict(model, X, va))
                P["tta"].append(predict(model, X, va, tta=args.tta))
                if vname == "xattn":
                    torch.manual_seed(1000 + seed)
                    P["kv_shuffle"].append(predict(model, X, va, kv_mode="shuffle"))
                    diags.append(diagnostics(model, X, va))
                per_seed.append(roc_auc_score(y[va, ti], P["plain"][-1]))
                eps.append(ep)
            if vname == "xattn" and scheme == "time":
                best_epochs[t] += eps
            row = dict(scheme=scheme, target=t, variant=vname, best_epoch_median=float(np.median(eps)),
                       auc_seed_mean=float(np.mean(per_seed)), auc_seed_std=float(np.std(per_seed)))
            for k, v in P.items():
                if v:
                    row[f"auc_{k}"] = float(roc_auc_score(y[va, ti], np.mean(v, 0)))  # seed 앙상블
            if diags:
                row.update(gate=float(np.mean([d["gate"] for d in diags])),
                           attn_entropy=float(np.mean([d["attn_entropy"] for d in diags])), ffn_depth=model.depth)
            results["runs"].append(row)
            results["val_preds"][f"{scheme}|{t}|{vname}"] = {"idx": va.tolist(), "y": y[va, ti].tolist(),
                                                              "p": np.mean(P["tta"], 0).tolist()}
            print(f"[{time.time() - t0:6.0f}s] {scheme:10s} {t} {vname:10s} "
                  + " ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)))

tag = "_full" if args.full else ""
if args.full:
    json.dump(results, open(ROOT / f"results/xattn{tag}_results.json", "w"), ensure_ascii=False, indent=1)
    tab = pd.DataFrame(results["runs"]).set_index(["scheme", "target", "variant"])
    print("\n== Validation AUC (full_raw, seed 앙상블)\n", tab[["auc_plain", "auc_tta", "auc_seed_mean", "auc_seed_std"]].round(3).to_string())
    raise SystemExit(0)

# 전체 train으로 재학습 → test(925~1124) 예측. epoch 수는 time split에서 고른 best epoch의 중앙값.
sub = {}
all_tr = np.where(is_train)[0]
for ti, t in enumerate(TARGETS):
    s = setup[t]
    X = prepare(s["streams"], all_tr, args.W, args.device)
    n_ep = max(int(np.median(best_epochs[t])), 5)
    preds = [predict(fit(lambda: VARIANTS["xattn"](s["dims"], s["kv"]), X, y_full[:, ti], all_tr, None, CFG, seed,
                         args.device, epochs=n_ep)[0], X, test_idx, tta=args.tta) for seed in range(args.seeds)]
    sub[f"pred_{ti + 1}"] = np.mean(preds, 0)
    results.setdefault("refit_epochs", {})[t] = n_ep

sub = pd.DataFrame({"Time_step": df.loc[test_idx, "Time_step"].to_numpy(), **sub})
out = ROOT / "submissions"
out.mkdir(exist_ok=True)
for name, n, ref in [(f"public_3반_안중범_{args.sub_idx}.csv", 100, "sample_submission_public.csv"),
                     ("private_3반_안중범.csv", 200, "sample_submission_private.csv")]:
    part = sub.iloc[:n]
    ref_ts = pd.read_csv(ROOT / ref)["Time_step"].to_numpy()
    assert len(part) == n and (part["Time_step"].to_numpy() == ref_ts).all()
    assert part[["pred_1", "pred_2"]].apply(lambda c: c.between(0, 1).all()).all()
    part.to_csv(out / name, index=False)
    print("saved", out / name)

json.dump(results, open(ROOT / "results/xattn_results.json", "w"), ensure_ascii=False, indent=1)
tab = pd.DataFrame(results["runs"])
cols = ["auc_plain", "auc_tta", "auc_kv_shuffle", "auc_seed_mean", "auc_seed_std", "gate", "attn_entropy"]
print("\n== Validation AUC (seed 앙상블)\n", tab.set_index(["scheme", "target", "variant"])[cols].round(3).to_string())
print(f"total {time.time() - t0:.0f}s")
