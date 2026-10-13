"""TabPFN 공용 헬퍼: .env 로드, 버전 고정 생성, MPS OOM fallback."""
import os

from .data import ROOT


def load_env():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            v = v.strip().strip("\"'")  # 따옴표 유무 모두 허용
            if v:
                os.environ.setdefault(k, v)


def make_model(version="v3.5", device="mps", **kw):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(
        ModelVersion(version), **{"device": device, "random_state": 0, "ignore_pretraining_limits": True, **kw})


def fit_predict(Xtr, ytr, Xte, version="v3.5", device="mps", log=print):
    """MPS 메모리 부족 시 memory_saving_mode → CPU 순으로 재시도."""
    for kw in ({}, {"memory_saving_mode": True}, {"device": "cpu"}):
        try:
            return make_model(version, **{"device": device, **kw}).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        except Exception as e:  # TabPFNMPSOutOfMemoryError 등
            if "out of memory" not in str(e).lower():
                raise
            log(f"  OOM → retry with {kw or 'memory_saving_mode'}")
    raise RuntimeError("TabPFN OOM on all fallbacks")
