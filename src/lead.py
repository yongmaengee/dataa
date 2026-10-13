"""Lead feature 세트 (전처리 동결 버전). 검증 근거: results/lead_val_*.json.

x[t+a] - x[t+b] 형태. 학습 행은 lead가 test 행을 참조하지 않도록 호출 측에서 마지막 MAX_LEAD행 제외.
"""
import numpy as np
import pandas as pd

from .ablation import make_features
from .data import group_columns
from .features import calendar

ALIGN = ((3, 0), (2, -1), (3, -1), (2, 0), (3, -2))  # 미국 지수 하루 시차 반영 anchor
MAX_LEAD = 3

FEATSETS = {  # 이름: (base 표현, lead 소스 그룹, anchor 목록)
    "fi_k3+z": ("zscore", "FI", ((3, 0),)),          # T1 TabPFN 최적 (wf5 0.805)
    "fi_al": (None, "FI", ALIGN),                     # T1 XGB 최적 (wf5 0.795)
    "all_k123+lc": ("level+change", "ALL", ((1, 0), (2, 0), (3, 0))),  # T2 최적 (wf5 0.982)
}


def leads(df, is_train, cols, anchors):
    X = df[cols]
    X = X.loc[:, X[is_train].std() > 0]
    return pd.concat([(X.shift(-a) - X.shift(-b)).add_suffix(f"_lead{a}_{b}") for a, b in anchors], axis=1)


def lead_frame(df, is_train, name, fit_mask):
    """DataFrame (전체 행). base 통계는 fit_mask 행으로만."""
    groups = group_columns(df)
    base, src, anchors = FEATSETS[name]
    cols = groups["FI"] if src == "FI" else [c for cs in groups.values() for c in cs]
    parts = [leads(df, is_train, cols, anchors)]
    if base:
        parts.insert(0, make_features(df, fit_mask, groups, repr=base, reduce="none"))
    else:
        parts.append(calendar(df))
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def streams(df, is_train, fit_mask):
    """cross-attention용 스트림: Q = FI lead(정렬 anchor) + FI 당일 표준화값, KV = 그룹별 zscore(KE/SH/WE)."""
    groups = group_columns(df)
    q = pd.concat([leads(df, is_train, groups["FI"], ALIGN), df[groups["FI"]].add_suffix("_lv"), calendar(df)], axis=1)
    kv = {g: pd.concat([make_features(df, fit_mask, {g: groups[g]}, repr="zscore", reduce="none", cal=False), calendar(df)], axis=1)
          for g in ["KE", "SH", "WE"]}
    return {"Q": q.replace([np.inf, -np.inf], np.nan), **{g: v.replace([np.inf, -np.inf], np.nan) for g, v in kv.items()}}
