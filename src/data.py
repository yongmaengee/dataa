"""데이터 로드, 그룹 정의, split 생성."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["Target_1", "Target_2"]
GROUP_PREFIX = {"FI": ("FI_",), "KE": ("KE_",), "SH": ("SH1_", "SH2_"), "WE": ("WE_",)}
SCHEMES = ["stratified", "random", "time"]


def load():
    """train + test_private(925~1124, test_public 포함)를 Time_step 순으로 결합."""
    tr = pd.read_csv(ROOT / "train.csv", encoding="cp949")
    te = pd.read_csv(ROOT / "test_private.csv", encoding="cp949")
    df = pd.concat([tr, te], ignore_index=True).sort_values("Time_step").reset_index(drop=True)
    is_train = df[TARGETS].notna().all(axis=1).to_numpy()
    return df, is_train


def group_columns(df):
    return {g: [c for c in df.columns if c.startswith(p)] for g, p in GROUP_PREFIX.items()}


def make_split(y, scheme, seed=0, val_frac=0.2, purge=3):
    """y: (n, 2) 이진 target. 반환: 정렬된 (train_idx, val_idx).

    stratified/random은 시간 순서를 섞으므로 인접일 자기상관 누수가 있음.
    time은 마지막 val_frac을 val로, 그 직전 purge일은 train에서 제외(D+3 target 겹침 방지).
    """
    n = len(y)
    idx = np.arange(n)
    if scheme == "stratified":
        strat = 2 * y[:, 0] + y[:, 1]
        tr, va = next(StratifiedShuffleSplit(1, test_size=val_frac, random_state=seed).split(idx, strat))
    elif scheme == "random":
        tr, va = next(ShuffleSplit(1, test_size=val_frac, random_state=seed).split(idx))
    elif scheme == "time":
        cut = int(round(n * (1 - val_frac)))
        tr, va = idx[: cut - purge], idx[cut:]
    else:
        raise ValueError(scheme)
    return np.sort(tr), np.sort(va)


def inner_split(y, tr, scheme, seed=0, frac=0.15, purge=3):
    """train fold 안에서 early stopping용 subset을 같은 scheme으로 분리."""
    a, b = make_split(y[tr], scheme, seed=seed + 100, val_frac=frac, purge=purge)
    return tr[a], tr[b]
