"""규정 2(평가 데이터 학습 사용 금지) 검증: test 행 feature를 노이즈로 교체해도 학습 입력 행렬이 동일한지 확인.
final_nokd.ipynb와 같은 feature 경로(src/lead.py, src/train.windows)를 사용. uv run python check_no_test_leak.py
"""
import numpy as np
from src.data import group_columns, load
from src.lead import MAX_LEAD, lead_frame, streams
from src.train import windows


def build(df, is_train):
    train_idx = np.where(is_train)[0]
    fit_idx = train_idx[:-MAX_LEAD]
    fm = np.zeros(len(df), bool)
    fm[fit_idx] = True
    out = {"T2_xgb": lead_frame(df, is_train, "all_k123+lc", fm).to_numpy(np.float32)[fit_idx],
           "T1_xgb": lead_frame(df, is_train, "fi_al", fm).fillna(0).to_numpy(np.float32)[fit_idx]}
    for g, A in streams(df, is_train, fm).items():
        A = A.fillna(0).to_numpy(np.float32)
        mu, sd = A[fm].mean(0), A[fm].std(0) + 1e-6
        out[f"T1_nn_{g}"] = windows(np.clip((A - mu) / sd, -5, 5), 10)[fit_idx]
    return out, fit_idx


if __name__ == "__main__":
    df, is_train = load()
    cols = [c for cs in group_columns(df).values() for c in cs]
    ref, fit_idx = build(df, is_train)
    print(f"학습 행 Time_step {df.Time_step[fit_idx[0]]}~{df.Time_step[fit_idx[-1]]} | test 행 {df.Time_step[~is_train].min()}~{df.Time_step[~is_train].max()}")
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(3):
        df2 = df.copy()
        df2.loc[~is_train, cols] = rng.standard_normal((int((~is_train).sum()), len(cols)))
        alt, _ = build(df2, is_train)
        same = {k: np.array_equal(np.nan_to_num(ref[k], nan=-999), np.nan_to_num(alt[k], nan=-999)) for k in ref}
        ok &= all(same.values())
        print(f"trial {trial}: {same}")
    print("PASS: 학습 입력은 test 행과 무관" if ok else "FAIL: 학습 입력이 test 행에 의존")
    raise SystemExit(0 if ok else 1)
