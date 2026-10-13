"""T1 기준 feature 조합별 manifold 진단 (모델 없이 데이터 기하만).

조합: level / change / zscore / change+zscore / tailored (모두 train 행 924개, train 통계로 표준화)
지표
  pca90           : 분산 90% 설명에 필요한 PC 수 (선형 차원)
  twonn_id        : TwoNN 내재 차원 (비선형 국소 차원)
  knn_auc         : k=15 이웃 label 평균의 AUC. |i-j|<=3 인접일은 이웃에서 제외(purge) → 모델 없는 예측 가능성
  knn_auc_umap    : UMAP 2D 공간에서 같은 지표
  time_locality   : 이웃 중 ±30일 이내 비율. 높을수록 manifold가 시간(regime)으로 조직됨 → 미래 일반화에 불리
  cka             : 조합 간 linear CKA (기하 유사도)
→ results/manifold_T1.png, results/manifold_T1.json
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.ablation import make_features
from src.data import ROOT, group_columns, load

FEATSETS = {
    "level": dict(repr="level", reduce="none"),
    "change": dict(repr="change", reduce="none"),
    "zscore": dict(repr="zscore", reduce="none"),
    "change+zscore": dict(repr="change+zscore", reduce="none"),
    "tailored": dict(repr="tailored"),
}
K, PURGE = 15, 3

df, is_train = load()
groups = group_columns(df)
y = df.loc[is_train, "Target_1"].to_numpy().astype(int)
n = len(y)
t_idx = np.arange(n)


def standardized(cfg):
    X = make_features(df, is_train, groups, **cfg)[is_train].fillna(0).to_numpy(np.float64)
    X = X[:, X.std(0) > 0]
    return (X - X.mean(0)) / X.std(0)


def pca90(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9) + 1)


def twonn(X):
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = np.sort(d[:, 2] / np.maximum(d[:, 1], 1e-12))
    F = np.arange(1, n + 1) / n
    keep = (F < 0.9) & (mu > 1)
    x, yy = np.log(mu[keep]), -np.log(1 - F[keep])
    return float((x * yy).sum() / (x * x).sum())


def knn_metrics(X):
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) if X.shape[1] <= 64 else \
        (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * X @ X.T
    D[np.abs(t_idx[:, None] - t_idx[None, :]) <= PURGE] = np.inf  # 자기 자신 + 인접일 제외
    nb = np.argsort(D, axis=1)[:, :K]
    score = y[nb].mean(1)
    return float(roc_auc_score(y, score)), float((np.abs(nb - t_idx[:, None]) <= 30).mean())


def cka(X, Y):
    Kx, Ky = X @ X.T, Y @ Y.T
    H = np.eye(n) - 1 / n
    Kx, Ky = H @ Kx @ H, H @ Ky @ H
    return float((Kx * Ky).sum() / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()))


Xs = {name: standardized(cfg) for name, cfg in FEATSETS.items()}
rows, emb = {}, {}
for name, X in Xs.items():
    auc, tl = knn_metrics(X)
    E = umap.UMAP(n_neighbors=K, min_dist=0.1, random_state=0).fit_transform(X)
    auc_u, tl_u = knn_metrics(E)
    emb[name] = E
    rows[name] = dict(n_feat=X.shape[1], pca90=pca90(X), twonn_id=round(twonn(X), 1), knn_auc=round(auc, 3),
                      time_locality=round(tl, 3), knn_auc_umap=round(auc_u, 3), time_locality_umap=round(tl_u, 3))
    print(name, rows[name], flush=True)

names = list(Xs)
C = pd.DataFrame([[cka(Xs[a], Xs[b]) for b in names] for a in names], index=names, columns=names)
print("\n== linear CKA (조합 간 기하 유사도)\n", C.round(2).to_string())
tab = pd.DataFrame(rows).T
print("\n== manifold 지표 (T1)\n", tab.to_string())
print(f"\n참고: 무작위 기준 time_locality ≈ {(61 - 2 * PURGE - 1) / (n - 2 * PURGE - 1):.3f}")

fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 7.5))
for j, name in enumerate(names):
    E = emb[name]
    axes[0, j].scatter(E[:, 0], E[:, 1], c=y, cmap="coolwarm", s=6, alpha=0.7)
    axes[0, j].set_title(f"{name}\nby Target_1  kNN-AUC={rows[name]['knn_auc']:.3f}", fontsize=9)
    sc = axes[1, j].scatter(E[:, 0], E[:, 1], c=t_idx, cmap="viridis", s=6, alpha=0.7)
    axes[1, j].set_title(f"by time  locality={rows[name]['time_locality']:.2f}", fontsize=9)
    for ax in axes[:, j]:
        ax.set_xticks([]), ax.set_yticks([])
fig.colorbar(sc, ax=axes[1, :], shrink=0.6, label="Time_step index")
fig.suptitle("UMAP of T1 feature sets (train rows)", fontsize=11)
fig.savefig(ROOT / "results/manifold_T1.png", dpi=130, bbox_inches="tight")
json.dump({"metrics": rows, "cka": C.round(4).to_dict(), "k": K, "purge": PURGE},
          open(ROOT / "results/manifold_T1.json", "w"), indent=1)
print("saved results/manifold_T1.png, results/manifold_T1.json")
