"""윈도우 텐서 생성, 학습(early stopping / 고정 epoch), 예측(TTA), 진단 지표."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def windows(A, W):
    """(N, F) -> (N, W, F). i번째 윈도우 = i-W+1..i 행 (과거 방향). 앞부분은 첫 행 반복으로 패딩."""
    pad = np.concatenate([np.repeat(A[:1], W - 1, axis=0), A], axis=0)
    return np.lib.stride_tricks.sliding_window_view(pad, W, axis=0).transpose(0, 2, 1).copy()


def prepare(streams, fit_idx, W, device, clip=5.0):
    """스트림별로 fit_idx 행의 통계로 표준화 → 윈도우화 → device 텐서."""
    out = {}
    for g, A in streams.items():
        A = np.asarray(A, dtype=np.float32)
        mu, sd = A[fit_idx].mean(0), A[fit_idx].std(0) + 1e-6
        Z = np.clip((A - mu) / sd, -clip, clip)
        out[g] = torch.from_numpy(windows(Z, W)).to(device)
    return out


def _augment(xb, cfg):
    out = {}
    for g, x in xb.items():
        x = x + cfg["noise"] * torch.randn_like(x)
        x = x * (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > cfg["feat_drop"])
        if g != "Q":  # KV 시점 토큰 dropout
            x = x * (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > cfg["tok_drop"])
        out[g] = x
    return out


def _take(X, idx):
    return {g: x[idx] for g, x in X.items()}


@torch.no_grad()
def predict(model, X, idx, tta=0, sigma=0.05, kv_mode="normal", bs=512):
    """tta=0: eval 모드 1회. tta=K: MC dropout(train 모드) + 입력 노이즈 K회 평균."""
    idx_t = torch.as_tensor(idx, device=next(model.parameters()).device)
    passes = max(tta, 1)
    model.train(tta > 0)
    probs = torch.zeros(len(idx), device=idx_t.device)
    for _ in range(passes):
        for s in range(0, len(idx), bs):
            xb = _take(X, idx_t[s:s + bs])
            if tta:
                xb = {g: x + sigma * torch.randn_like(x) for g, x in xb.items()}
            probs[s:s + bs] += torch.sigmoid(model(xb, kv_mode)[0])
    model.eval()
    return (probs / passes).cpu().numpy()


@torch.no_grad()
def diagnostics(model, X, idx):
    """gate 값(tanh)과 정규화 attention 엔트로피(1 = 균등 분포, 0 = 한 토큰에 집중)."""
    if not getattr(model, "use_cross", False):
        return {}
    model.eval()
    idx_t = torch.as_tensor(idx, device=next(model.parameters()).device)
    _, w = model(_take(X, idx_t))
    ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1) / math.log(w.shape[-1])
    return {"gate": float(torch.tanh(model.xattn.gate)), "attn_entropy": float(ent.mean())}


def fit(make_model, X, y, tr, es, cfg, seed, device, epochs=None):
    """es가 있으면 es AUC로 early stopping, 없으면 epochs만큼 고정 학습. 반환: (model, best_epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    n_ep = epochs if epochs is not None else cfg["max_epochs"]
    best, best_ep, best_state, wait = -1.0, n_ep, None, 0
    for ep in range(1, n_ep + 1):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), cfg["bs"]):
            bi = torch.as_tensor(perm[s:s + cfg["bs"]], device=device)
            logits, _ = model(_augment(_take(X, bi), cfg))
            loss = F.binary_cross_entropy_with_logits(logits, yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if es is None:
            continue
        auc = roc_auc_score(y[es], predict(model, X, es))
        if auc > best:
            best, best_ep, best_state, wait = auc, ep, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep
