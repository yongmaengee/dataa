"""서브셋/전처리 ablation용 feature 생성 + walk-forward split.

모델과 평가는 고정하고 feature 쪽만 바꾸기 위한 모듈. 축소(군집/PCA) 통계는 fold의 train 행으로만 적합.
"""
import numpy as np
import pandas as pd

from . import collinearity as co
from .data import GROUP_PREFIX
from .features import _rz, build, calendar, tabular

REPRS = {
    "level": ("level",),
    "change": ("d1", "d3", "d7"),
    "zscore": ("z7", "z28"),
    "level+change": ("level", "d1", "d3", "d7"),
    "change+zscore": ("d1", "d3", "d7", "z7", "z28"),
    "all": ("level", "d1", "d3", "d7", "z7", "z28"),
}


def walk_forward(n, n_folds=5, val=92, purge=3):
    """확장형 walk-forward: 마지막 n_folds*val 행을 val 블록으로 나누고, 각 블록 이전 전체(purge 제외)를 train으로."""
    start = n - n_folds * val
    return [(np.arange(0, start + k * val - purge), np.arange(start + k * val, start + (k + 1) * val))
            for k in range(n_folds)]


def _represent(base, kinds):
    parts = []
    for k in kinds:
        if k == "level":
            parts.append(base)
        elif k[0] == "d":
            parts.append(base.diff(int(k[1:])).add_suffix(f"_{k}"))
        elif k[0] == "z":
            parts.append(_rz(base, int(k[1:])).add_suffix(f"_{k}"))
    return pd.concat(parts, axis=1)


def _reduce(df, cols, fit_mask, mode, g, thr=0.85, pca_above=32):
    if mode == "none":
        return df[[c for c in cols if df.loc[fit_mask, c].std() > 0]]
    kept, _ = co.cluster_reduce(df.loc[fit_mask, cols], thr)
    if mode == "cluster+pca" and len(kept) > pca_above:
        return co.pca_reduce(df[kept], fit_mask, g)[0]
    return df[kept]


def make_features(df, fit_mask, groups, repr="all", reduce="cluster+pca", include=None, cal=True,
                  domain=False, extras=()):
    """repr: REPRS 키 또는 'tailored'(기존 그룹별 맞춤 파생변수). include: 사용할 그룹 목록(None=전체).
    domain: 그룹별 도메인 파생변수를 같은 표현으로 추가. extras: 그대로 이어 붙일 DataFrame들(EXAONE, 전문가 확률 등).
    """
    include = list(groups) if include is None else include
    if repr == "tailored":
        feats, cal_df, _ = build(df, fit_mask, {g: groups[g] for g in include})
        X = tabular(feats, cal_df) if cal else pd.concat(feats.values(), axis=1)
    else:
        parts = [_represent(_reduce(df, groups[g], fit_mask, reduce, g), REPRS[repr]) for g in include]
        if domain:
            dx = domain_extras(df)
            dcols = [c for c in dx.columns if any(c.startswith(p) for g in include for p in GROUP_PREFIX[g])]
            if dcols:
                parts.append(_represent(dx[dcols], REPRS[repr]))
        if cal:
            parts.append(calendar(df))
        X = pd.concat(parts, axis=1)
    if len(extras):
        X = pd.concat([X, *extras], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


PRESET = {  # run_ablation.py Stage 1에서 확정한 target별 표현
    "Target_1": dict(repr="change+zscore", reduce="none"),
    "Target_2": dict(repr="level+change", reduce="none"),
}


def domain_extras(df):
    """그룹별 도메인 파생변수(원시 형태). 표현 변환은 make_features에서 그룹과 같은 방식으로 적용.

    컬럼 접두사(FI_/WE_)로 소속 그룹을 판단.
    """
    ang = 2 * np.pi * df["WE_최대풍속풍향_deg_"]  # [0,1] 스케일 → 0~360도 선형 매핑 가정
    return pd.DataFrame({
        "FI_sp_brent_wti": df["FI_Brent"] - df["FI_WTI"],
        "FI_sp_term": df["FI_Corp3yr"] - df["FI_CD91"],
        "FI_sp_gas_diesel": df["FI_Gasoline"] - df["FI_Diesel"],
        "WE_temp_range": df["WE_최고기온"] - df["WE_최저기온"],
        "WE_wind_u": df["WE_최대풍속_m_s_"] * np.sin(ang),
        "WE_wind_v": df["WE_최대풍속_m_s_"] * np.cos(ang),
        "WE_rain": (df["WE_강수량"] > 0).astype(float),
        "WE_pm10_ma7": df["WE_PM10_미세먼지_"].rolling(7, min_periods=1).mean(),
        "WE_pm25_ma7": df["WE_PM2_5_초미세먼지_"].rolling(7, min_periods=1).mean(),
    })


def exaone_surprise(df, ex, h=3):
    """EXAONE surprise: h일 전 예측 대비 실제 변화의 초과분. t 시점에 모두 관측 가능(causal).

    surp_t  = (x_t - x_{t-h}) - ex_d3_{t-h}
    surpz_t = surp_t / (q90-q10)_{t-h}   (예측 불확실성으로 정규화)
    """
    out = {}
    for c in ex.columns:
        if c.endswith("_ex_d3"):
            b = c[: -len("_ex_d3")]
            surp = (df[b] - df[b].shift(h)) - ex[c].shift(h)
            out[f"{b}_ex_surp"] = surp
            out[f"{b}_ex_surpz"] = surp / (ex[f"{b}_ex_unc"].shift(h) + 1e-6)
    return pd.DataFrame(out)
