"""results/*.json → report.html (단일 파일, manifold 그림 내장). uv run python make_report.py"""
import base64
import html
import json

import numpy as np

from src.data import ROOT

R = lambda f: json.load(open(ROOT / "results" / f, encoding="utf-8"))
final, abl, xa, xf, tp = R("final.json"), R("ablation.json"), R("xattn_results.json"), R("xattn_full_results.json"), R("tabpfn_results.json")
conv, exa, jerk, mani, coll, exp = R("conv2d_T1.json"), R("exaone_t1.json"), R("jerk_T1.json"), R("manifold_T1.json"), R("collinearity.json"), R("tabpfn_experts.json")
png = base64.b64encode(open(ROOT / "results/manifold_T1.png", "rb").read()).decode()
T = ["Target_1", "Target_2"]
E = html.escape
f3 = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
pm = lambda m, s: f"{m:.3f} <span class=se>±{s:.3f}</span>"

# baselines.ipynb 단일 holdout XGB 결과 (노트북 출력에서 옮김; 나머지 모델은 JSON에서 읽음)
XGB_HOLDOUT = {"time": {"subset": (.558, .908), "full_raw": (.450, .931), "subset+exaone": (.476, .934), "full_raw+exaone": (.554, .933)},
               "stratified": {"subset": (.649, .960), "full_raw": (.661, .956), "subset+exaone": (.666, .966), "full_raw+exaone": (.667, .964)},
               "random": {"subset": (.580, .940), "full_raw": (.668, .946), "subset+exaone": (.587, .945), "full_raw+exaone": (.691, .957)}}


def table(head, rows, cls="", note=None):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<p class=note>{note}</p>" if note else ""
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>{cap}'


def dotchart(items, lo, hi, ref=None, best=None, w=720):
    """items: [(label, mean, se)] 정렬된 순. 가로 점 + ±se 막대, 세로 기준선."""
    lh, pad_l, pad_r, top = 26, 230, 24, 28
    h = top + lh * len(items) + 34
    x = lambda v: pad_l + (v - lo) / (hi - lo) * (w - pad_l - pad_r)
    ticks = np.round(np.arange(lo, hi + 1e-9, (hi - lo) / 5), 3)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="AUC 비교">']
    for t in ticks:
        s.append(f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{h - 30}" class="grid"/>'
                 f'<text x="{x(t):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{t:.2f}</text>')
    if ref is not None:
        s.append(f'<line x1="{x(ref):.1f}" y1="{top - 10}" x2="{x(ref):.1f}" y2="{h - 30}" class="ref"/>'
                 f'<text x="{x(ref) + 4:.1f}" y="{top - 12}" class="tick">{ref:.2f} = 무작위</text>')
    for i, (lab, m, se) in enumerate(items):
        cy = top + lh * i + lh / 2
        c = "mark best" if lab == best else "mark"
        s.append(f'<text x="{pad_l - 10}" y="{cy + 4}" class="lab" text-anchor="end">{E(lab)}</text>')
        if se:
            s.append(f'<line x1="{x(m - se):.1f}" y1="{cy}" x2="{x(m + se):.1f}" y2="{cy}" class="err"/>')
        s.append(f'<circle cx="{x(m):.1f}" cy="{cy}" r="5" class="{c}"><title>{E(lab)}: {m:.3f}' + (f" ± {se:.3f}" if se else "") + '</title></circle>'
                 f'<text x="{x(m) + 9:.1f}" y="{cy + 4}" class="val">{m:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- 공통 데이터
ft = {t: {r["model"]: r for r in final["tables"][t]} for t in T}
s1 = {r["config"]: r for r in abl["stage1"]}
runs_xa = {(r["scheme"], r["target"], r["variant"]): r for r in xa["runs"]}
runs_xf = {(r["scheme"], r["target"], r["variant"]): r for r in xf["runs"]}
runs_tp = {(r["scheme"], r["target"], r["featset"]): r["auc"] for r in tp["runs"]}
folds_txt = ", ".join(f"{a}–{b}" for a, b in final["folds"])


def holdout_table(ti):
    rows = []
    for name, src in [("XGBoost", None), ("TabPFN v3.5", "tp"), ("concat_mlp", "concat_mlp"), ("q_only (단일 스트림 transformer)", "q_only"), ("xattn (group cross-attention)", "xattn")]:
        for fs in ["subset", "full_raw", "subset+exaone", "full_raw+exaone"]:
            cells = []
            for sc in ["time", "stratified", "random"]:
                if src is None:
                    v = XGB_HOLDOUT[sc][fs][ti]
                elif src == "tp":
                    v = runs_tp.get((sc, T[ti], fs))
                else:
                    if fs not in ("subset", "full_raw"):
                        v = None
                    elif fs == "subset":
                        v = runs_xa.get((sc, T[ti], src), {}).get("auc_tta")
                    else:
                        v = runs_xf.get((sc, T[ti], src), {}).get("auc_tta")
                cells.append(f3(v))
            if all(c == "–" for c in cells):
                continue
            rows.append([f"{name} [{fs}]", *cells])
    return table(["모델 [feature 세트]", "time (purge 3일)", "stratified", "random"], rows, "num",
                 "단일 holdout(val 20%). stratified·random은 인접일 누수로 낙관적. subset = 공선성 축소+그룹별 파생(초기 버전), full_raw = 원본 284개, +exaone = EXAONE 예측 feature 추가.")


def xattn_diag(ti):
    rows = []
    for sc in ["time", "stratified", "random"]:
        r = runs_xa[(sc, T[ti], "xattn")]
        rows.append([sc, f3(r["auc_plain"]), f3(r["auc_kv_shuffle"]), f"{r['auc_plain'] - r['auc_kv_shuffle']:+.3f}", f"{r['gate']:+.3f}", f3(r["attn_entropy"])])
    return table(["split", "xattn AUC", "KV 셔플 AUC", "KV 효과", "gate tanh", "attention 엔트로피"], rows, "num",
                 "KV 셔플: 학습된 모델에 다른 샘플의 KV를 넣어 평가. gate≈0, 엔트로피≈1이면 cross-attention이 사실상 꺼져 있음.")


def stage1_table(ti):
    t = T[ti]
    rows = []
    for name, r in s1.items():
        m, d, se = r[f"{t}_mean"], r[f"{t}_diff_vs_ref"], r[f"{t}_diff_se"]
        rows.append([E(name), r["n_feat"], f3(m), f"{d:+.3f} ± {se:.3f}" if name != "level / none" else "기준", f3(min(r[f"{t}_folds"]))])
    return table(["표현 / 축소", "feature 수", "AUC", "원본 대비 (paired)", "fold 최저"], rows, "num",
                 "XGBoost 고정(300 trees, depth 3), walk-forward 5 fold, 3 seed. 기준 = level / none (원본 그대로).")


def stage2_table(ti):
    t = T[ti]
    rows = [[E(r["config"]), r["n_feat"], f3(r[f"{t}_mean"]), f"{r[f'{t}_diff_vs_ref']:+.3f} ± {r[f'{t}_diff_se']:.3f}"] for r in abl[f"stage2_{t}"]]
    rows += [[E(r["config"]), r["n_feat"], f3(r[f"{t}_mean"]), f"{r[f'{t}_diff_vs_ref']:+.3f} ± {r[f'{t}_diff_se']:.3f}"] for r in abl[f"stage3_{t}"][1:]]
    return table(["설정", "feature 수", "AUC", "전체 대비"], rows, "num")


def final_table(ti):
    t = T[ti]
    rows = []
    for name, r in ft[t].items():
        rows.append([("<b>" + E(name) + "</b>") if name == final["chosen"][t] else E(name), pm(r["mean"], r["se"]),
                     "기준" if name == "xgb[base]" else f"{r['diff_vs_base']:+.3f} ± {r['diff_se']:.3f}", f3(r["min_fold"])])
    return table(["후보", "AUC (5 fold)", "xgb[base] 대비", "fold 최저"], rows, "num",
                 "base = 확정 표현. domain = 스프레드·풍향 벡터·일교차·강수·미세먼지 7일 평균. exaone = EXAONE 예측변화량·불확실성·surprise. experts = TabPFN 그룹 전문가 OOF 확률(stacking).")


def experts_table(ti):
    t = T[ti]
    rows = [[g, f3(np.mean(exp["block_auc"][f"{t}|{g}"][-5:])), ", ".join(f"{v:.2f}" for v in exp["block_auc"][f"{t}|{g}"][-5:])] for g in ["FI", "KE", "SH", "WE", "all"]]
    return table(["전문가 (입력 그룹)", "AUC (마지막 5 블록)", "블록별"], rows, "num", "TabPFN v3.5, 각 92일 블록은 블록 시작 3일 전까지의 행으로만 fit(causal OOF).")


# ---------- 세션 2: lead feature
LV = {m: R(f"lead_val_{m}.json") for m in ["xgb", "tabpfn"]}
NL, NLK, NL10 = R("neural_lead_T1.json")["runs"], R("neural_lead_T1_kd.json")["runs"], R("neural_lead_T1_wf10.json")["runs"]
def _lv(m, key, sc): r = LV[m].get(key, {}).get(sc); return f3(r["mean"]) if r else "–"
def _lvmin(m, key): r = LV[m].get(key, {}).get("wf5"); return f3(min(r["folds"])) if r else "–"
def _s100(m, key): r = LV[m].get(key, {}).get("sliding100"); return f"{r['min']:.2f} / {r['median']:.2f} / {r['max']:.2f}" if r else "–"
lead_rows = []
for m, key, label in [("tabpfn", "Target_1|fi_k3+z", "TabPFN · zscore + FI t+3 lead"), ("tabpfn", "Target_1|fi_k123+z", "TabPFN · zscore + FI lead k=1..3"),
                      ("xgb", "Target_1|fi_al", "XGB · FI 정렬 lead 90개 + 요일 (base 없음)"), ("xgb", "Target_1|all_k123+lc", "XGB · level+change + 전체 lead"),
                      ("xgb", "Target_1|fi_k123+cz", "XGB · change+zscore + FI lead k=1..3"), ("tabpfn", "Target_1|fx_k123", "TabPFN · 환율 3개 lead만 (9개)")]:
    lead_rows.append([label, LV[m].get(key, {}).get("n_feat", "–"), _lv(m, key, "wf5"), _lv(m, key, "wf10"), _lv(m, key, "time20"), _lvmin(m, key), _s100(m, key)])
lead_rows_t2 = [[label, LV["xgb"][key]["n_feat"], _lv("xgb", key, "wf5"), _lv("xgb", key, "wf10"), _lv("xgb", key, "time20"), _lvmin("xgb", key)]
                for key, label in [("Target_2|all_k123+lc", "XGB · level+change + 전체 lead k=1..3 (최종)"), ("Target_2|all_k123+cz", "XGB · change+zscore + 전체 lead"), ("Target_2|fi_k123+cz", "XGB · FI lead만")]]
nn_rows = [[k, pm(NL[k]["mean"], NL[k]["se"]), f3(min(NL[k]["folds"])), f3(NL10[k]["mean"]) if k in NL10 else "–", f3(NL10[k]["train_auc"]) if k in NL10 else "–",
            pm(NLK[k]["mean"], NLK[k]["se"]), f"{NL[k]['n_params']:,}", f"{NL[k]['gate_mean']:+.3f}" if "gate_mean" in NL[k] else "–"] for k in NL]
# 블렌드 (저장된 OOF로 재계산)
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from src.data import load as _load
_df, _it = _load(); _y = _df.loc[_it, "Target_1"].to_numpy().astype(int)
def _blend(nl, xgb_oof, folds):
    idx = np.array(nl["conv1d"]["oof"]["idx"]); pn = np.array(nl["conv1d"]["oof"]["p"]); px = xgb_oof[idx]
    fa = lambda p: np.array([roc_auc_score(_y[f], p[np.isin(idx, f)]) for f in folds])
    b = fa((rankdata(pn) + rankdata(px)) / 2); return b.mean(), b.std(ddof=1) / np.sqrt(len(b)), b.min()
from src.ablation import walk_forward as _wf
_n = int(_it.sum())
bl5 = _blend(NL, np.load(ROOT / "results/xgb_fi_al_oof_wf5.npy"), [va for _, va in _wf(_n)])
bl10 = _blend(NL10, np.load(ROOT / "results/xgb_fi_al_oof_wf10.npy"), [va for _, va in _wf(_n, 10, 46)])
blend_rows = [["conv1d + XGB fi_al 순위평균 (최종)", pm(bl5[0], bl5[1]), f3(bl5[2]), pm(bl10[0], bl10[1]), f3(bl10[2])],
              ["TabPFN fi_k3+z (참고, 사전학습 모델)", _lv("tabpfn", "Target_1|fi_k3+z", "wf5"), _lvmin("tabpfn", "Target_1|fi_k3+z"), _lv("tabpfn", "Target_1|fi_k3+z", "wf10"), "–"],
              ["XGB fi_al 단독", _lv("xgb", "Target_1|fi_al", "wf5"), _lvmin("xgb", "Target_1|fi_al"), _lv("xgb", "Target_1|fi_al", "wf10"), "–"]]

# ---------- T1 전용
t1_items = [("tabpfn[base]", ft["Target_1"]["tabpfn[base]"]["mean"], ft["Target_1"]["tabpfn[base]"]["se"]),
            ("xgb[base]", ft["Target_1"]["xgb[base]"]["mean"], ft["Target_1"]["xgb[base]"]["se"]),
            ("xgb[base+domain]", ft["Target_1"]["xgb[base+domain]"]["mean"], ft["Target_1"]["xgb[base+domain]"]["se"]),
            ("xgb[base+domain+experts]", ft["Target_1"]["xgb[base+domain+experts]"]["mean"], ft["Target_1"]["xgb[base+domain+experts]"]["se"]),
            ("xgb[base+domain+exaone]", ft["Target_1"]["xgb[base+domain+exaone]"]["mean"], ft["Target_1"]["xgb[base+domain+exaone]"]["se"]),
            ("conv1d (시간축)", conv["runs"]["conv1d"]["mean"], conv["runs"]["conv1d"]["se"]),
            ("conv2d + xattn + FFN", conv["runs"]["conv2d_xattn"]["mean"], conv["runs"]["conv2d_xattn"]["se"]),
            ("EXAONE P(rise) 18개 → LR", exa["stack"]["LR on 18 FI P(rise)"]["mean"], exa["stack"]["LR on 18 FI P(rise)"]["se"]),
            ("conv2d", conv["runs"]["conv2d"]["mean"], conv["runs"]["conv2d"]["se"]),
            ("xgb[d1+d2+d3 미분]", jerk["xgb_walkforward"]["d1+d2+d3"]["mean"], jerk["xgb_walkforward"]["d1+d2+d3"]["se"]),
            ("xgb[tailored 초기 subset]", s1["tailored (기존 subset)"]["Target_1_mean"], None),
            ("mlp_flat (윈도우 MLP)", conv["runs"]["mlp_flat"]["mean"], conv["runs"]["mlp_flat"]["se"]),
            ("xgb[level / 원본 그대로]", s1["level / none"]["Target_1_mean"], None)]
t1_items.sort(key=lambda r: -r[1])
t2_items = [(k, r["mean"], r["se"]) for k, r in ft["Target_2"].items()] + \
           [("xgb[level / 원본 그대로]", s1["level / none"]["Target_2_mean"], None), ("xgb[tailored 초기 subset]", s1["tailored (기존 subset)"]["Target_2_mean"], None),
            ("xgb[change+zscore]", s1["change+zscore / none"]["Target_2_mean"], None)]
t2_items.sort(key=lambda r: -r[1])

conv_rows = [[k, pm(r["mean"], r["se"]), ", ".join(f"{v:.2f}" for v in r["folds"]), f"{r['n_params']:,}", f"{r['gate_mean']:+.3f}" if "gate_mean" in r else "–"] for k, r in conv["runs"].items()]
jerk_rows = [[f"d{k}", f"{jerk['structure'][f'd{k}']['lag1_autocorr_median']:+.3f}", f"{jerk['reference']['random walk 과차분 서명'][f'd{k}']:+.3f}",
              f"{jerk['structure'][f'd{k}']['var_ratio_vs_d1_median']:.2f}", f"{jerk['single_feature_signal'][f'd{k}']:.3f}"] for k in [1, 2, 3]]
jerk_x = [[k, r["n_feat"], pm(r["mean"], r["se"])] for k, r in jerk["xgb_walkforward"].items()]
mani_rows = [[k, m["n_feat"], m["pca90"], m["twonn_id"], f3(m["knn_auc"]), f3(m["time_locality"])] for k, m in mani["metrics"].items()]
zs = sorted(exa["zero_shot"].items(), key=lambda kv: -kv[1]["mean"])
exa_rows = [[k.replace("_prise", ""), f3(v["mean"]), ", ".join(f"{x:.2f}" for x in v["folds"])] for k, v in zs[:4] + [kv for kv in zs if kv[0] in ("US3_mean", "LABEL_prise")] + zs[-2:]]
coll_rows = [[g, r["n_raw"], r["n_cluster"], r.get("n_pca", "–"), f"{coll['target_relevance']['Target_1'][g]['top5_mean']:.2f}", f"{coll['target_relevance']['Target_2'][g]['top5_mean']:.2f}"] for g, r in coll["within"].items()]
cca = coll["cca_diff7"]
cca_rows = [[a] + [f"{cca[a][b]:.2f}" for b in cca] for a in cca]

css = """
:root{--bg:#f6f7f9;--surface:#ffffff;--fg:#1b2130;--muted:#5d6675;--line:#d8dde5;--accent:#1f6fb0;--accent2:#d2532a;--soft:#e8f0f8;
 --font:"IBM Plex Sans KR","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#13171d;--surface:#1a2029;--fg:#e4e8ee;--muted:#98a2b1;--line:#2b333f;--accent:#4a94cc;--accent2:#dc6f45;--soft:#1f2a38;color-scheme:dark}}
:root[data-theme=dark]{--bg:#13171d;--surface:#1a2029;--fg:#e4e8ee;--muted:#98a2b1;--line:#2b333f;--accent:#4a94cc;--accent2:#dc6f45;--soft:#1f2a38;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--font);font-size:15px;line-height:1.65}
.wrap{max-width:1040px;overflow-wrap:anywhere;margin:0 auto;padding-block:32px 64px;padding-inline:20px}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 6px;text-wrap:balance}h2{font-size:1.4rem;margin:56px 0 12px;padding-top:16px;border-top:2px solid var(--fg);text-wrap:balance}
h3{font-size:1.08rem;margin:30px 0 8px}p{max-width:72ch;margin:8px 0}ul{max-width:72ch;padding-left:1.2em}li{margin:3px 0}
.eyebrow{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}
.lead{color:var(--muted);max-width:72ch}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:18px 0 8px;font-size:.9rem}nav a{color:var(--accent);text-decoration:none}nav a:hover,nav a:focus-visible{text-decoration:underline}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));gap:12px;margin:22px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px}.tile .k{font-size:.78rem;color:var(--muted);letter-spacing:.04em}
.tile .v{font-family:var(--mono);font-size:1.6rem;font-variant-numeric:tabular-nums;margin:2px 0}.tile .s{font-size:.85rem;color:var(--muted)}
.tw{overflow-x:auto;margin:10px 0 4px}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.8rem;letter-spacing:.03em;white-space:nowrap}table.num td:not(:first-child){font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.se{color:var(--muted);font-size:.82em}.note{font-size:.82rem;color:var(--muted);max-width:none;margin:4px 0 0}
.callout{background:var(--soft);border-left:3px solid var(--accent);padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0;max-width:none}.callout p{max-width:none}
.verdict{display:inline-block;font-size:.75rem;padding:1px 8px;border-radius:10px;border:1px solid var(--line);color:var(--muted);margin-left:6px;vertical-align:middle;font-family:var(--mono)}
.chart{width:100%;height:auto;max-width:100%;display:block;margin:12px 0 4px}.chart .grid{stroke:var(--line);stroke-width:1}.chart .ref{stroke:var(--accent2);stroke-width:1.5;stroke-dasharray:4 3}
.chart .tick,.chart .lab,.chart .val{fill:var(--muted);font-family:var(--mono);font-size:11px}.chart .lab{fill:var(--fg);font-family:var(--font);font-size:12px}
.chart .err{stroke:var(--accent);stroke-width:2;stroke-linecap:round;opacity:.55}.chart .mark{fill:var(--accent);stroke:var(--surface);stroke-width:2}.chart .mark.best{fill:var(--accent2)}
img{max-width:100%;border:1px solid var(--line);border-radius:4px}code{font-family:var(--mono);font-size:.85em;background:var(--soft);padding:1px 5px;border-radius:3px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:16px}.two>*{min-width:0}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
"""

doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>KOSPI·건강식품 D+3 예측 실험 보고서</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style></head><body><div class="wrap">
<p class="eyebrow">데이터분석 미니프로젝트 · Multi-target classification · 평가지표 AUC</p>
<h1>KOSPI·건강식품 D+3 상승 예측 실험 보고서</h1>
<p class="lead">train 924일(Time_step 1–924), 설명변수 284개(FI 금융 18 · KE 검색어 51 · SH 쇼핑 199 · WE 기상 15 · 요일). Target_1 = D+3 KOSPI 상승, Target_2 = D+3 건강식품 지수 상승. 이 문서는 시도한 모든 방법을 target별로 나눠 정리하고, 같은 검증 체계(walk-forward 5 fold)에서 비교한 결과를 담는다.</p>
<nav><a href="#setup">검증 체계</a><a href="#t1">Target_1 (causal)</a><a href="#t2">Target_2 (causal)</a><a href="#lead">Lead feature와 최종 파이프라인</a><a href="#final">최종 선택</a><a href="#files">재현</a></nav>

<div class="tiles">
<div class="tile"><div class="k">Target_1 · 최종 (lead feature)</div><div class="v">{bl5[0]:.3f}</div><div class="s">1D conv + XGBoost 순위평균 · wf10 {bl10[0]:.3f} · fold 최저 {bl5[2]:.3f}</div></div>
<div class="tile"><div class="k">Target_2 · 최종 (lead feature)</div><div class="v">{LV['xgb']['Target_2|all_k123+lc']['wf5']['mean']:.3f}</div><div class="s">XGBoost · level+change + 전체 lead · fold 최저 {_lvmin('xgb', 'Target_2|all_k123+lc')}</div></div>
<div class="tile"><div class="k">causal 제약 시 T1 상한</div><div class="v">{ft['Target_1']['tabpfn[base]']['mean']:.3f}</div><div class="s">과거 feature만 사용 · 11개 방법 모두 0.59 이하</div></div>
<div class="tile"><div class="k">사전학습 모델 의존</div><div class="v">0</div><div class="s">최종 파이프라인은 XGBoost + 자체 학습 1D conv · 외부 데이터 없음</div></div>
</div>

<h2 id="setup">검증 체계와 공통 전처리</h2>
<h3>검증 방식이 바뀐 이유</h3>
<p>처음에는 단일 holdout 세 가지(time / stratified / random, val 20%)로 비교했다. stratified와 random은 시간 순서를 섞기 때문에 val 날짜의 인접일이 train에 들어가고, D+3 target의 자기상관(lag 1에서 0.47)과 rolling 파생변수를 통해 정보가 샌다. 그 결과 모든 모델이 time split보다 0.1–0.2 높은 점수를 냈고, 이 점수는 미래 구간(test)에 해당하지 않는다. 그래서 중반부터는 <b>walk-forward 5 fold</b>(val 92일씩: Time_step {folds_txt}, 각 fold는 val 시작 3일 전까지의 데이터로만 학습)를 기준으로 삼았다. 이후 모든 표의 AUC는 별도 표기가 없으면 이 5 fold의 평균 ± 표준오차다.</p>
<div class="callout"><p><b>읽는 법.</b> fold가 5개뿐이라 T1처럼 AUC 0.55–0.60 구간의 차이는 표준오차(0.01–0.02)의 1–2배 안에 들어오는 경우가 많다. "paired 차이 ± se"가 se의 2배를 넘지 않으면 같은 성능으로 읽는 것이 안전하다.</p></div>

<h3>그룹별 공선성 축소와 target 관련도</h3>
{table(["그룹", "원본", "군집 대표 (|corr|≥0.85)", "PCA (90%)", "T1 관련도", "T2 관련도"], coll_rows, "num", "관련도 = feature와 target의 Spearman |ρ| 상위 5개 평균. T2는 쇼핑(SH)·검색어(KE)와 0.7 수준으로 강하게 연결되고, T1은 어느 그룹도 0.18을 넘지 않는다.")}
<h3>그룹 간 정준상관 (7일 차분, 상위 5 PC)</h3>
{table(["", *cca.keys()], cca_rows, "num", "FI는 다른 그룹과 0.19–0.25로 보완적이고, KE–SH는 0.59로 서로 겹친다. 이 결과로 T1은 FI를 Q로, 나머지를 KV로 두는 cross-attention 구조를 설계했다.")}

<h2 id="t1">Target_1 · D+3 KOSPI 상승</h2>
<p class="lead">결론: 어떤 모델·전처리·외부 모델도 walk-forward AUC 0.59를 넘지 못했다. 전처리(수준값 제거, PCA 제거)가 0.45 → 0.58로 올린 것이 유일하게 뚜렷한 개선이고, 그 이후의 시도는 모두 노이즈 범위다. manifold·미분 진단은 label이 feature 공간의 국소 구조와 시간축 국소 패턴 어디에도 없음을 보여 준다.</p>

<h3>시도 전체 (walk-forward 5 fold)</h3>
{dotchart(t1_items, 0.42, 0.66, ref=0.5, best="tabpfn[base]")}
<p class="note">점 = 평균 AUC, 가로선 = ±표준오차(없는 항목은 ablation 3 seed 결과로 se 미계산). 주황 점 = 최종 선택.</p>

<h3>1. 전처리·서브셋 ablation <span class="verdict">XGB 고정</span></h3>
<p>모델(XGBoost 300 trees, depth 3)과 평가를 고정하고 표현 방식과 축소 방법만 바꿨다. 수준값(level)을 넣으면 5 fold 모두 0.5 미만으로 무작위보다 나쁘고, 변화량(d1·d3·d7)과 추세 대비 z-score(z7·z28)가 유효하다. PCA는 어떤 조합에서도 손해다. 처음에 만든 "tailored" subset이 낮았던 이유도 수준값과 PCA를 포함했기 때문이다.</p>
{stage1_table(0)}
<h3>그룹 포함 여부·EXAONE (T1 최적 표현 zscore/none 기준)</h3>
{stage2_table(0)}
<p>그룹을 하나 빼서 유의하게 좋아지는 경우는 없고(FI 제외 +0.021 ± 0.022), WE 제외는 일관되게 손해다. 따라서 전체 그룹을 유지한다. 최종 표현은 zscore와 평균이 같으면서 fold 최저값이 더 높은 <b>change+zscore / none</b>(1,412개)으로 확정했다.</p>

<h3>2. 종합 후보 비교와 최종 선택 <span class="verdict">5 seed</span></h3>
{final_table(0)}
<p>도메인 파생변수, EXAONE 파생변수(예측 변화량·불확실성·surprise), TabPFN 그룹 전문가 stacking 모두 base보다 낮다. TabPFN은 XGB보다 +0.008 ± 0.025로 차이가 유의하지 않지만, fold 최저가 0.560으로 가장 안정적이라 선택했다.</p>
<h3>TabPFN 그룹 전문가 (stacking 재료)</h3>
{experts_table(0)}
<p>어느 그룹 하나로도 0.57을 넘지 못한다. 정보가 거의 없는 확률을 feature로 더했으니 stacking이 손해였던 것은 당연한 결과다.</p>

<h3>3. Group-wise cross-attention <span class="verdict">단일 holdout</span></h3>
<p>FI 스트림(과거 20일 토큰)을 Q, KE·SH·WE 스트림을 KV로 두고 self-attn → gated cross-attn → FFN(깊이는 KV 크기에 비례)으로 구성했다. 대조군은 cross-attn을 뺀 q_only와 당일 feature concat MLP다. 이 실험은 초기 subset(수준값·PCA 포함)과 단일 holdout으로 수행했으므로 절대값은 위 표와 직접 비교할 수 없다.</p>
{xattn_diag(0)}
<p>random split에서 KV를 셔플하면 0.741 → 0.603으로 크게 떨어지는데, 이것은 KV가 유용한 정보라서가 아니라 인접일 누수를 KV가 운반하고 있었다는 뜻이다. time split에서는 xattn(0.536)이 q_only(0.583)보다 낮고 gate도 0.08에 머문다.</p>

<h3>4. 단일 holdout 모델 비교 (초기 단계) <span class="verdict">참고</span></h3>
{holdout_table(0)}
<p>"전체 데이터를 한 번에"(full_raw) 넣으면 time split에서 네 모델 모두 subset보다 낮다. 원본 값은 전체 기간 min-max 스케일의 수준값이라 regime 변화에 일반화되지 않는다. 반대로 random split에서는 full_raw가 더 높은데, 수준값이 "어느 시기인지"를 알려 줘 누수를 더 잘 이용하기 때문이다.</p>

<h3>5. 1D / 2D conv와 cross-attention <span class="verdict">3 seed</span></h3>
<p>입력은 원본 284 계열의 과거 20일 윈도우(시점 = 행, feature = 열)를 윈도우별로 instance norm한 것이다. 2D conv의 feature 축 인접성을 위해 열은 그룹 → 상관 군집 순서로 정렬했다.</p>
{table(["모델", "AUC", "fold별", "파라미터", "gate"], conv_rows, "num")}
<p>conv1d와 conv2d+xattn이 평균 0.56대로 XGB에 근접하지만 표준오차가 3–5배 크고, 같은 모델이 fold에 따라 0.36과 0.75를 오간다. feature 축을 섞는 2D conv는 1D보다 오히려 낮고, cross-attention gate는 0.002로 닫혀 있다.</p>

<h3>6. 시간축 미분 진단 (1D conv가 의미 있는가)</h3>
<div class="two"><div>{table(["차수", "lag-1 자기상관 (중앙값)", "random-walk 과차분 서명", "분산비 vs d1", "단일 feature |AUC−0.5| 상위5"], jerk_rows, "num")}</div>
<div>{table(["XGB 입력", "feature 수", "AUC"], jerk_x, "num")}</div></div>
<p>2차·3차 미분의 자기상관(−0.31, −0.52)이 random-walk를 과차분했을 때의 서명(−0.50, −0.67)에 가깝다. 즉 고차 미분은 새 정보가 아니라 증폭된 노이즈이고, 단일 feature 신호도 차수가 오를수록 줄어든다. d1의 lag 0–3을 모두 넣어도 d1 하나와 같다. 시간축 커널이 뽑을 국소 미분 구조에 T1 정보가 없다는 뜻이며, 유효했던 z-score는 미분이 아니라 20–28일 추세 대비 위치였다.</p>

<h3>7. Manifold 진단 (feature 조합별 기하)</h3>
{table(["조합", "feature 수", "PCA 90% 차원", "TwoNN 내재 차원", "kNN-AUC (인접 3일 제외)", "time locality"], mani_rows, "num", "kNN-AUC = 이웃 15개의 label 평균으로 만든 AUC(모델 없는 예측 가능성). time locality = 이웃 중 ±30일 이내 비율, 무작위 기준 0.06.")}
<img src="data:image/png;base64,{png}" alt="feature 조합별 UMAP. 위 행은 Target_1로, 아래 행은 시간으로 색칠">
<p>level과 tailored의 manifold는 시간 순서로 이어진 고리 모양이고 label은 그 위에 무작위로 섞여 있다(locality 0.89, 0.63). 이 구조에서 "비슷한 행"은 곧 "인접한 날짜"라서 random split 점수가 부풀고 미래 구간에서 무너진다. change·zscore는 시간 구조를 지우지만(0.10–0.20) label로도 구분되지 않는다(kNN-AUC 0.51–0.52). 전처리는 누수를 없애는 역할은 했지만 T1 신호 자체가 국소 기하에 거의 없다.</p>

<h3>8. EXAONE Forecast for Finance 1.0 (zero-shot 직접 분류)</h3>
<p>금융 계열 18개를 각 날짜까지의 값(최근 512일)으로 3일 뒤 분위수 예측을 만들고, P(x<sub>t+3</sub> &gt; x<sub>t</sub>)를 상승 확률로 썼다. T1 label 계열은 t−3까지만 알 수 있으므로 context를 거기서 끊었다.</p>
{table(["계열", "zero-shot AUC", "fold별"], exa_rows, "num", "18개 계열의 AUC가 0.40–0.58에 고르게 퍼져 평균은 0.5 부근이다. 상위 계열은 다중 비교의 우연 범위이고, KOSPI와 가장 관련 있어야 할 미국 3지수 평균은 0.437로 반대 방향이다.")}
<p>18개 P(rise)를 로지스틱 회귀로 묶으면 {pm(exa['stack']['LR on 18 FI P(rise)']['mean'], exa['stack']['LR on 18 FI P(rise)']['se'])}. EXAONE 예측값을 XGB feature로 넣는 방식(2번)도 −0.035였다. 이 모델은 계열을 독립 채널로 예측하므로 KOSPI와의 교차 관계를 쓰지 못한다.</p>

<h2 id="t2">Target_2 · D+3 건강식품 지수 상승</h2>
<p class="lead">결론: 어떤 설정에서도 0.96–0.97이 나오는 포화 상태다. 쇼핑·검색어 feature가 target과 0.7 수준으로 직접 연결돼 있어 모델·전처리 선택이 거의 영향을 주지 않는다. TabPFN이 가장 안정적이라 선택했다.</p>
<h3>시도 전체 (walk-forward 5 fold)</h3>
{dotchart(t2_items, 0.94, 0.98, best="tabpfn[base]")}
<h3>1. 전처리·서브셋 ablation <span class="verdict">XGB 고정</span></h3>
{stage1_table(1)}
<p>PCA를 넣은 설정만 0.954–0.960으로 약간 낮고, 나머지는 0.960–0.968로 차이가 없다. T1과 달리 수준값(level)이 유용하므로 <b>level+change / none</b>(1,130개)으로 확정했다.</p>
<h3>그룹 포함 여부·EXAONE (level+change 기준)</h3>
{stage2_table(1)}
<p>SH 또는 KE만으로도 0.96이고, WE만으로는 0.919, FI만으로는 0.940이다. 전체를 쓸 때 가장 높다.</p>
<h3>2. 종합 후보 비교와 최종 선택 <span class="verdict">5 seed</span></h3>
{final_table(1)}
{experts_table(1)}
<h3>3. Group-wise cross-attention <span class="verdict">단일 holdout</span></h3>
<p>SH 스트림을 Q, FI·KE·WE를 KV로 두었다. KV를 셔플해도 AUC가 변하지 않고 gate는 0 근처, 엔트로피는 0.97로 균등 분포다. 모델이 KV를 쓰지 않는다.</p>
{xattn_diag(1)}
<h3>4. 단일 holdout 모델 비교 (초기 단계) <span class="verdict">참고</span></h3>
{holdout_table(1)}

<h2 id="lead">Lead feature와 최종 파이프라인</h2>
<p class="lead">위 두 절은 모든 feature를 과거 방향(causal)으로만 만든 결과다. 그런데 이 과제는 배치 채점이라 test 파일에 925~1124일의 feature가 전부 주어지고, 규정(25쪽)은 "평가용 데이터를 <b>학습</b>에 활용 금지"와 "외부 데이터 금지"만 명시한다. t일의 label은 KOSPI의 t→t+3 변화이므로, 같은 구간의 환율·해외지수 변화(t+1~t+3일 feature, 이하 lead)를 예측 <b>입력</b>으로 쓰면 T1이 0.59에서 0.80으로 오른다. 리더보드 상위권(0.9 내외)도 이 정보를 쓴 것으로 추정된다.</p>
<div class="callout"><p><b>규정 준수 방식.</b> (1) 학습 행은 1~921만 사용: 922~924행의 lead는 test 행을 참조하므로 학습에서 제외. (2) 예측 시 test 행 간 참조는 학습이 아니라 입력. (3) 사전학습 모델(TabPFN·EXAONE)은 "외부 데이터" 해석 리스크와 재현성(토큰·라이선스) 문제로 최종 파이프라인에서 제외하고 비교 실험에만 사용. (4) 13쪽의 leakage 정의("예측 시점에 알 수 없는 정보")에는 lead가 해당할 수 있으므로, 이 보고서는 causal 결과와 lead 결과를 분리해 제시한다.</p></div>
<h3>단일 feature 신호</h3>
{table(["feature", "구간", "AUC"], [["USD/KRW", "t+3 − t", "0.739 (역방향 0.261)"], ["JPY/KRW", "t+3 − t", "0.730 (역방향)"], ["S&P500", "(t+2) − (t−1)", "0.715"], ["Nasdaq", "(t+2) − (t−1)", "0.713"], ["S&P500", "t+3 − t", "0.681"], ["Brent", "(t+3) − (t−2)", "0.623"], ["KE·SH·WE 최고", "t+3 − t", "≤ 0.551"]], "num", "미국 지수는 한국 장 마감 후에 움직이므로 하루 시차를 둔 구간이 최적. 환율은 같은 시간대라 t+3 − t 그대로. 검색어·쇼핑·기상의 lead에는 T1 신호가 없다.")}
<h3>Target_1 후보 × 검증 방식</h3>
{table(["모델 · feature", "n_feat", "wf5", "wf10", "time20", "wf5 fold 최저", "100일 슬라이딩 min / med / max"], lead_rows, "num", "wf10 = 46일 × 10 fold. 모든 후보가 Time_step 741~832 구간에서 0.55~0.65로 약하다. 100일 public 점수는 구간에 따라 0.64~0.88 사이로 예상되며, 실제 public 리더보드도 이 범위였다.")}
<h3>Target_2 후보</h3>
{table(["모델 · feature", "n_feat", "wf5", "wf10", "time20", "wf5 fold 최저"], lead_rows_t2, "num", "level+change + 전체 lead가 세 검증 모두 1위이고, causal TabPFN(0.972)보다 5개 fold 전부에서 높다.")}
<h3>Lead 입력 신경망: 1D conv · cross-attention · KD</h3>
<p>전처리는 동결하고 기법만 바꿨다. 공통으로 pre-LayerNorm residual, dropout 0.3, weight decay 2e-2, 가우시안 노이즈·feature dropout·시점 dropout 증강, inner-tail early stopping, 5 seed 평균. 입력은 FI 정렬 lead 90개 + FI 당일값 + 요일의 과거 10일 윈도우이고, cross-attention은 KE/SH/WE zscore 윈도우를 KV로 받는다. KD는 TabPFN(fi_k3+z)의 causal OOF 확률을 teacher로 둔 변형이다.</p>
{table(["모델", "wf5 AUC", "fold 최저", "wf10 AUC", "train AUC (wf10)", "KD 버전 wf5", "파라미터", "gate"], nn_rows, "num", "네 구조의 OOF 예측 상관은 0.88~0.94로 사실상 같은 모델이다. cross-attention gate는 0.02로 lead 입력에서도 KV 정보를 쓰지 않는다. train 0.86 vs val 0.79로 과적합 폭 0.07. KD는 네 구조 모두에서 비-KD보다 낮다.")}
<h3>블렌드와 최종 선택</h3>
{table(["구성", "wf5", "wf5 fold 최저", "wf10", "wf10 fold 최저"], blend_rows, "num", "1D conv와 XGB의 OOF 상관이 0.84라 서로 다른 오차를 상쇄한다. 선택 규칙은 미리 정했다: wf5·wf10·fold 최저 세 기준에서 일관되게 앞서는 구성만 채택.")}
<p>최종 Target_1은 <b>1D conv + XGBoost(fi_al) 순위평균</b>이다. TabPFN 단독과 평균이 같으면서 약한 구간의 최저값이 더 높고, 사전학습 모델 없이 train.csv와 test_private.csv만으로 재현된다. 예측 확률의 인접일 평활(t±1 평균)은 효과가 없었다(±0.006).</p>

<h2 id="final">최종 선택과 남은 과제</h2>
{table(["target", "표현", "축소", "그룹", "모델", "walk-forward AUC"], [
 ["Target_1", "FI 정렬 lead 90 + FI 당일값 + 요일 (신경망) / FI 정렬 lead + 요일 (XGB)", "없음", "FI + 요일", "1D conv + XGBoost 순위평균", pm(bl5[0], bl5[1])],
 ["Target_2", "level + change + 전체 lead (k=1..3)", "없음", "전체 + 요일", "XGBoost", pm(LV['xgb']['Target_2|all_k123+lc']['wf5']['mean'], LV['xgb']['Target_2|all_k123+lc']['wf5']['se'])]], "")}
<ul>
<li><b>제출 파일</b>: <code>final_nokd.ipynb</code>가 train.csv와 test_private.csv만으로 <code>submissions/private_3반_안중범_nokd.csv</code>(200행)를 생성한다. 행 수·Time_step·[0,1] 범위·결측을 검사한다. causal 버전(<code>private_3반_안중범_tabpfn_causal.csv</code>, T1 0.587)도 보관한다.</li>
<li><b>T1 해석</b>: causal 제약에서는 데이터에 신호가 없고(manifold·미분 진단), lead를 허용하면 환율·미국 지수의 동시 구간 변화가 대부분의 신호를 준다. 모델 선택의 영향은 ±0.01로 작다.</li>
<li><b>선택 편향</b>: target당 여러 후보 중 최고를 골랐으므로 절대값은 약간 낙관적이다. 수준값 제외·PCA 제외·lead 도입처럼 차이가 큰 결론은 이 편향으로 뒤집히지 않는다.</li>
<li><b>재현성</b>: 시드 고정, 패키지 버전 출력, git 관리. MPS는 완전 결정적이지 않아 소수점 셋째 자리 차이가 날 수 있다. macOS에서 xgboost는 torch보다 먼저 import해야 한다(libomp 충돌).</li>
<li><b>하지 않은 것</b>: 블렌드 가중치·윈도우 길이·dropout 민감도, 외부 데이터(규정상 금지).</li>
</ul>

<h2 id="files">재현</h2>
<p>환경: <code>uv sync</code> (Python 3.12, torch 2.14 MPS, tabpfn 9.2.0 · 모델 v3.5, exaone-forecast). <code>.env</code>에 <code>TABPFN_TOKEN</code>(Prior Labs 라이선스 동의 필요)과 <code>HF_TOKEN</code>.</p>
{table(["스크립트", "내용", "출력"], [
 ["run_ablation.py", "표현×축소×그룹 ablation (XGB 고정, walk-forward)", "results/ablation.json"],
 ["run_exaone_feats.py", "EXAONE Finance rolling 3일 예측 feature (causal)", "results/exaone_feats.csv"],
 ["run_tabpfn_experts.py", "TabPFN 그룹 전문가 causal OOF 확률", "results/tabpfn_experts.csv"],
 ["run_final.py", "종합 후보 비교, 최종 선택, 제출 파일", "results/final.json, submissions/"],
 ["run_xattn.py [--full]", "group cross-attention (단일 holdout 3종)", "results/xattn_results.json"],
 ["run_tabpfn.py", "TabPFN feature 세트별 단일 holdout", "results/tabpfn_results.json"],
 ["run_manifold.py / run_jerk.py", "T1 manifold·시간축 미분 진단", "results/manifold_T1.*, jerk_T1.json"],
 ["run_conv2d.py / run_exaone_t1.py", "T1 1D·2D conv, EXAONE zero-shot 분류", "results/conv2d_T1.json, exaone_t1.json"],
 ["baselines.ipynb", "단일 holdout 비교표·ROC (XGB는 노트북 안에서 실행)", "—"],
 ["src/lead.py · run_lead_val.py", "lead feature 세트, 후보 × 검증(wf5/wf10/time20)", "results/lead_val_*.json"],
 ["src/neural_lead.py · run_neural_lead.py", "lead 입력 ResMLP / ResConv1D / LeadXAttn, --kd, --scheme wf10", "results/neural_lead_T1*.json"],
 ["final_nokd.ipynb / final_kd.ipynb", "Private 제출 재현 (사전학습 모델 없음 / KD 비상용)", "submissions/*_nokd.csv"],
 ["make_report.py", "이 보고서 생성", "report.html"]], "",
 "TabPFN과 XGBoost를 같은 프로세스에서 쓰면 macOS에서 libomp 충돌로 멈추므로 TabPFN은 항상 별도 스크립트로 실행한다.")}
</div></body></html>"""

open(ROOT / "report.html", "w", encoding="utf-8").write(doc)
print("saved report.html", f"{len(doc) / 1e6:.2f} MB")
