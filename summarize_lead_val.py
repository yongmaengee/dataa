"""results/lead_val_{xgb,tabpfn}.json → 후보×검증 표 + TabPFN·XGB 블렌드(wf5 OOF 순위평균)."""
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from src.data import ROOT, TARGETS, load
pd.set_option("display.width", 220)
df, is_train = load(); y = df.loc[is_train, TARGETS].to_numpy().astype(int)
R = {m: json.load(open(ROOT / f"results/lead_val_{m}.json")) for m in ["xgb", "tabpfn"] if (ROOT / f"results/lead_val_{m}.json").exists()}
f = lambda r, k: f"{r[k]['mean']:.3f}" + (f"±{r[k]['se']:.3f}" if r[k].get("se") else "") if k in r else "–"
for t in TARGETS:
    rows = []
    for m, res in R.items():
        for key, r in res.items():
            if not key.startswith(t) or "wf5" not in r: continue
            s = r.get("sliding100", {})
            rows.append(dict(model=m, cand=key.split("|")[1], n_feat=r.get("n_feat"), wf5=f(r, "wf5"), wf5_min=f"{min(r['wf5']['folds']):.3f}", wf10=f(r, "wf10"), time20=f(r, "time20"), random=f(r, "random"),
                             s100_min=f"{s.get('min', np.nan):.3f}", s100_med=f"{s.get('median', np.nan):.3f}", s100_max=f"{s.get('max', np.nan):.3f}"))
    print(f"\n== {t}\n", pd.DataFrame(rows).sort_values("wf5", ascending=False).to_string(index=False))
    if len(R) == 2:  # 블렌드: 두 모델 모두 OOF가 있는 후보
        ti = TARGETS.index(t); print("\n-- blend (wf5 OOF 순위평균, TabPFN+XGB)")
        for key in R["tabpfn"]:
            if not key.startswith(t) or "oof_wf5" not in R["tabpfn"][key] or key not in R["xgb"] or "oof_wf5" not in R["xgb"][key]: continue
            a, b = R["tabpfn"][key]["oof_wf5"], R["xgb"][key]["oof_wf5"]; idx = np.array(a["idx"]); pa, pb = np.array(a["p"]), np.array(b["p"])
            folds = [idx[(idx >= s) & (idx < s + 92)] for s in range(464, 924, 92)]
            def auc(p): return [roc_auc_score(y[fi, ti], p[np.isin(idx, fi)]) for fi in folds]
            bl = (rankdata(pa) + rankdata(pb)) / 2
            print(f"  {key.split('|')[1]:14s} tabpfn={np.mean(auc(pa)):.3f} xgb={np.mean(auc(pb)):.3f} blend={np.mean(auc(bl)):.3f}  blend folds={np.round(auc(bl),3).tolist()}")
