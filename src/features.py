"""그룹별 파생변수(= 그룹별 inductive bias). 모든 연산은 과거 방향(causal).

원본 값은 이미 [0,1] min-max 스케일이고 0 근처 값이 많아 변화율 대신 차분을 사용.
"""
import numpy as np
import pandas as pd


def _rz(d, w):
    """rolling z-score: 최근 w일 추세 대비 현재 값의 이탈 정도."""
    m = d.rolling(w, min_periods=5).mean()
    s = d.rolling(w, min_periods=5).std()
    return (d - m) / (s + 1e-6)


def _fi(df, base):
    sp = pd.DataFrame({
        "FI_sp_brent_wti": df["FI_Brent"] - df["FI_WTI"],
        "FI_sp_term": df["FI_Corp3yr"] - df["FI_CD91"],
        "FI_sp_gas_diesel": df["FI_Gasoline"] - df["FI_Diesel"],
    })
    base = pd.concat([base, sp], axis=1)
    return pd.concat([
        base,
        base.diff(1).add_suffix("_d1"),   # 단기 모멘텀
        base.diff(3).add_suffix("_d3"),   # target horizon과 같은 3일
        base.diff(5).add_suffix("_d5"),
        _rz(base, 20).add_suffix("_z20"),  # 1개월 추세 대비 이탈
    ], axis=1)


def _trend(base):
    """검색어/쇼핑 지수: 주간 주기가 강하므로 7일 차분 + 4주 rolling z-score."""
    return pd.concat([base, base.diff(7).add_suffix("_d7"), _rz(base, 28).add_suffix("_z28")], axis=1)


def _we(df, base):
    # 풍향은 [0,1] 스케일이므로 0~360도를 선형 매핑했다고 가정
    ang = 2 * np.pi * df["WE_최대풍속풍향_deg_"]
    extra = pd.DataFrame({
        "WE_temp_range": df["WE_최고기온"] - df["WE_최저기온"],
        "WE_wind_u": df["WE_최대풍속_m_s_"] * np.sin(ang),
        "WE_wind_v": df["WE_최대풍속_m_s_"] * np.cos(ang),
        "WE_rain": (df["WE_강수량"] > 0).astype(float),
        "WE_pm10_ma7": df["WE_PM10_미세먼지_"].rolling(7, min_periods=1).mean(),
        "WE_pm25_ma7": df["WE_PM2_5_초미세먼지_"].rolling(7, min_periods=1).mean(),
    })
    return pd.concat([base, extra, base.diff(1).add_suffix("_d1")], axis=1)


def calendar(df):
    a = 2 * np.pi * (df["Weekday"] - 1) / 7
    return pd.DataFrame({"CAL_sin": np.sin(a), "CAL_cos": np.cos(a)})


def derive(df, bases):
    """bases: 공선성 축소(군집 대표 또는 PC) 후 그룹별 base DataFrame. 반환: 그룹 -> 파생 feature DataFrame."""
    feats = {
        "FI": _fi(df, bases["FI"]),
        "KE": _trend(bases["KE"]),
        "SH": _trend(bases["SH"]),
        "WE": _we(df, bases["WE"]),
    }
    return {g: X.replace([np.inf, -np.inf], np.nan).fillna(0.0) for g, X in feats.items()}


def tabular(feats, cal):
    """베이스라인(XGB/TabPFN)용: 당일 시점의 모든 그룹 feature를 한 행으로."""
    return pd.concat([*feats.values(), cal], axis=1)


def build(df, is_train, groups, thr=0.85, pca_above=32):
    """공선성 축소 → 파생변수까지 한 번에. 통계는 train 행(1~924)으로만 적합.

    반환: (feats: 그룹 -> DataFrame, cal: DataFrame, report: dict)
    """
    from . import collinearity as co

    trn = df[is_train]
    bases, report = {}, {"within": {}}
    for g, cols in groups.items():
        kept, clusters = co.cluster_reduce(trn[cols], thr)
        info = {"n_raw": len(cols), "n_cluster": len(kept), "vif_after_cluster": co.max_vif(trn[kept])}
        if len(kept) > pca_above:
            bases[g], ev = co.pca_reduce(df[kept], is_train, g)
            info.update(n_pca=bases[g].shape[1], pca_var=ev)
        else:
            bases[g] = df[kept]
        info["clusters"] = {k: v for k, v in clusters.items() if len(v) > 1}
        report["within"][g] = info
    feats = derive(df, bases)
    trb = {g: b[is_train] for g, b in bases.items()}
    report["cca_level"] = co.cca_matrix(trb).round(3).to_dict()
    report["cca_diff7"] = co.cca_matrix(trb, diff=7).round(3).to_dict()
    y = trn[["Target_1", "Target_2"]].to_numpy().astype(int)
    report["target_relevance"] = {
        t: co.target_relevance({g: X[is_train].reset_index(drop=True) for g, X in feats.items()}, y[:, i])
        for i, t in enumerate(["Target_1", "Target_2"])
    }
    return feats, calendar(df), report


def featsets(df, is_train, groups):
    """베이스라인 비교용 feature 세트.

    subset          : 공선성 축소 + 그룹별 파생변수 (cross-attention 실험과 동일)
    full_raw        : 원본 feature 전체 + calendar (가공 없음)
    *+exaone        : 위 세트에 EXAONE Finance rolling 예측 feature(results/exaone_feats.csv) 추가
    """
    feats, cal, _ = build(df, is_train, groups)
    raw = pd.concat([df[[c for cols in groups.values() for c in cols]], cal], axis=1)
    sets = {"subset": tabular(feats, cal), "full_raw": raw}
    from .data import ROOT
    p = ROOT / "results/exaone_feats.csv"
    if p.exists():
        ex = pd.read_csv(p, encoding="utf-8")
        assert (ex["Time_step"].to_numpy() == df["Time_step"].to_numpy()).all()
        ex = ex.drop(columns="Time_step")
        sets["subset+exaone"] = pd.concat([sets["subset"], ex], axis=1)
        sets["full_raw+exaone"] = pd.concat([raw, ex], axis=1)
    return sets
