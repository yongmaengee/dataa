"""그룹 내부 공선성 축소(계층 군집) + 그룹 간 정준상관(CCA) + target 관련도."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cluster_reduce(X, thr=0.85, lag=7):
    """|corr|이 thr 이상인 변수들을 하나의 군집으로 묶고 군집별 medoid 1개만 남김.

    수준(level) 상관은 공통 추세에 지배되므로 lag일 차분 기준으로 상관을 계산.
    """
    X = X.loc[:, X.std() > 0]
    Xd = X.diff(lag).iloc[lag:]
    Xd = Xd.loc[:, Xd.std() > 0]
    C = Xd.corr().abs().fillna(0).to_numpy()
    cols = Xd.columns.to_numpy()
    if len(cols) == 1:
        return list(cols), {cols[0]: list(cols)}
    D = 1 - C
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), 1 - thr, "distance")
    kept, clusters = [], {}
    for k in np.unique(lab):
        m = np.where(lab == k)[0]
        rep = cols[m[np.argmax(C[np.ix_(m, m)].mean(1))]]
        kept.append(rep)
        clusters[rep] = list(cols[m])
    return kept, clusters


def pca_reduce(X_all, train_mask, prefix, var=0.90, n_max=24):
    """군집 축소 후에도 변수가 많은 그룹용. train 행으로만 표준화·PCA를 적합하고 전체 행에 적용."""
    A = X_all.to_numpy()
    mu, sd = A[train_mask].mean(0), A[train_mask].std(0) + 1e-9
    Z = (A - mu) / sd
    _, s, Vt = np.linalg.svd(Z[train_mask], full_matrices=False)
    ev = np.cumsum(s**2) / np.sum(s**2)
    k = int(min(np.searchsorted(ev, var) + 1, n_max))
    pcs = pd.DataFrame(Z @ Vt[:k].T, index=X_all.index, columns=[f"{prefix}_pc{i}" for i in range(k)])
    return pcs, float(ev[k - 1])


def max_vif(X):
    X = X.loc[:, X.std() > 0]
    R = np.corrcoef(X.to_numpy(), rowvar=False)
    return float(np.max(np.diag(np.linalg.pinv(R))))


def _orth_pcs(A, k):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, : min(k, A.shape[1])]


def cancorr(A, B, k=5):
    """두 그룹의 상위 k개 PC 사이 첫 정준상관. k를 제한해 표본 수 대비 과적합(상관 부풀림)을 억제."""
    Qa, Qb = _orth_pcs(A, k), _orth_pcs(B, k)
    return float(np.linalg.svd(Qa.T @ Qb, compute_uv=False)[0])


def cca_matrix(groups, k=5, diff=None):
    names = list(groups)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    arrs = {}
    for g, X in groups.items():
        Z = X.diff(diff).iloc[diff:] if diff else X
        arrs[g] = Z.loc[:, Z.std() > 0].to_numpy()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            M.loc[a, b] = M.loc[b, a] = cancorr(arrs[a], arrs[b], k)
    return M


def target_relevance(feats, y):
    """그룹별 feature와 target의 Spearman |rho| 최대/상위5 평균."""
    out = {}
    for g, X in feats.items():
        r = X.apply(lambda c: c.rank()).corrwith(pd.Series(y), method="pearson").abs().fillna(0)
        out[g] = {"max": float(r.max()), "top5_mean": float(r.nlargest(5).mean())}
    return out
