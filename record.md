# record.md — validation 기록 · 회고 · 다음 접근

상세 표와 그림은 `report.html`(`uv run python make_report.py`로 재생성), 원본 수치는 `results/*.json`.

---

## 2026-10-13 · 세션 1

### 검증 설계 (중간에 바꿈)
- 초기: 단일 holdout 3종 (time / stratified / random, val 20%). stratified·random은 인접일 누수로 time 대비 +0.1~0.2 부풀림 확인 → **폐기**.
- 확정: **walk-forward 5 fold**, val 92일씩 (Time_step 465–556, 557–648, 649–740, 741–832, 833–924), 각 fold는 val 시작 3일 전까지만 학습(purge). 이하 수치는 모두 이 기준, 평균 ± 표준오차.
- 판정 기준: paired 차이가 se의 2배 안이면 동률로 본다. fold 5개라 T1 0.55–0.60 구간은 대부분 동률.

### 측정 결과 — Target_1 (D+3 KOSPI)

| 시도 | AUC | 비고 |
|---|---|---|
| **TabPFN v3.5 [change+zscore]** | **0.587 ± 0.013** | fold 최저 0.560 · 최종 선택 |
| XGB [change+zscore] | 0.580 ± 0.019 | fold 최저 0.530 |
| XGB + 도메인 파생변수 | 0.564 | −0.016 ± 0.013 |
| XGB + EXAONE 파생(d3·unc·surprise) | 0.545 | −0.035 ± 0.022 |
| XGB + TabPFN 그룹전문가 stacking | 0.554 | −0.026 ± 0.013 |
| conv1d (시간축) | 0.565 ± 0.075 | fold 0.36–0.75, 불안정 |
| conv2d + xattn + FFN | 0.563 ± 0.059 | gate 0.002 (닫힘) |
| conv2d | 0.524 ± 0.062 | |
| mlp_flat (윈도우 MLP) | 0.471 ± 0.040 | |
| EXAONE P(rise) 18개 → LR | 0.552 ± 0.041 | zero-shot 단일 계열 최고 0.584(Nikel), 미국3지수 0.437 |
| XGB [d1+d2+d3 미분] | 0.517 ± 0.020 | |
| XGB [tailored 초기 subset] | 0.508 | level+PCA 포함이 원인 |
| XGB [level / 원본 그대로] | 0.448 | 5 fold 모두 <0.5 |

전처리 ablation 요점 (XGB 고정): level 제거 +0.13, PCA는 모든 조합에서 −0.05~−0.08, 군집 축소는 무영향. 그룹 제외는 유의한 이득 없음(FI 제외 +0.021 ± 0.022, WE 제외 −0.018 ± 0.006).

### 측정 결과 — Target_2 (D+3 건강식품)

| 시도 | AUC |
|---|---|
| **TabPFN v3.5 [level+change]** | **0.972 ± 0.004** (fold 최저 0.957, 최종 선택) |
| XGB [level+change] | 0.968 ± 0.007 |
| XGB + domain / + exaone / + experts | 0.964–0.968 (무영향) |
| 그룹 단독: SH 0.963, KE 0.960, FI 0.940, WE 0.919 |
| level / 원본 그대로 | 0.960 |

포화 상태. 모델·전처리 선택이 거의 무의미하고 PCA만 소폭 손해.

### 단일 holdout 초기 결과 (참고, 재사용 금지)
- group cross-attention (FI→Q, KE·SH·WE→KV): time T1 0.536 (q_only 0.583, concat_mlp 0.581), T2 0.919. KV 셔플 시 random split에서 0.741→0.603 급락 = KV가 인접일 누수를 운반. gate ≤0.08, 엔트로피 ≥0.91.
- "전체 데이터 한 번에"(full_raw): time split에서 4개 모델 모두 subset보다 낮음 (XGB 0.558→0.450, TabPFN 0.590→0.505). random에서는 반대로 높음(누수 활용).

### 진단
- **manifold (UMAP, kNN-AUC 인접3일 제외)**: level·tailored는 시간으로 조직된 고리(time locality 0.89 / 0.63, 무작위 0.06), label은 무작위 분포. change·zscore는 시간 구조 제거(0.10–0.20)되지만 kNN-AUC 0.51–0.52로 label 구조도 없음.
- **시간축 미분**: d2·d3 lag-1 자기상관 −0.31 / −0.52 ≈ random-walk 과차분 서명(−0.50 / −0.67). 분산 1.9× / 5.0×. 고차 미분은 노이즈 증폭. 1D conv가 뽑을 국소 패턴 없음.
- target 관련도: T2는 SH·KE와 |ρ| 0.7, T1은 어느 그룹도 0.18 미만.

### 회고
- **잘된 것**: 검증을 walk-forward로 바꾼 뒤 모든 결론이 일관됨. level 제거·PCA 제거는 효과 크기가 se의 4–9배라 선택 편향으로 뒤집히지 않음. target별 표현 분리(T1 change+zscore, T2 level+change)는 유지.
- **안 된 것**: cross-attention(2종 구조), 도메인 파생변수, EXAONE(feature·직접 분류 모두), TabPFN stacking, conv 계열. T1은 모든 시도가 0.58–0.59 천장. 모델이 아니라 신호 부재로 판단.
- **cross-attention 실패 원인(사후)**: KV 그룹들이 T1에 대해 개별로도 0.47–0.57(정보 없음) → gate가 0 근처에서 열리지 않음. 초기 실험은 level+PCA feature와 단일 holdout이라 walk-forward 재검증 안 됨.
- **삽질/수정**: MPS SDPA가 attention dropout 미지원 → 0으로. TabPFN+XGB 같은 프로세스에서 libomp 데드락 → TabPFN은 항상 별도 스크립트. `.env` 값에 따옴표 → 파서가 strip하도록. TabPFN v3.5는 Prior Labs 라이선스 별도 동의 필요. T1 1412 feature는 TabPFN MPS OOM → memory_saving → CPU fallback(전문가 생성 24분). MPS adaptive pool 크기 제약 → 고정 풀링. 단일 holdout 하나로 모델 고르려다 노이즈에 끌려다님.
- **선택 편향 주의**: target당 후보 6개 중 최고 선택. 절대값 약간 낙관적.

### 제출 상태
- `submissions/public_3반_안중범_3.csv`, `submissions/private_3반_안중범.csv` = TabPFN(확정 표현, 전체 train fit). 행 수·Time_step·범위 검사 완료.
- public 제출 2회차는 xattn 기반이었음(권장 안 함). private는 1회만 가능 → 위 파일로.

### 다음 접근 구상
1. **TabPFN + XGB 확률 평균 앙상블** (T1 분산 축소 목적, 같은 5 fold로 검증, 수 분).
2. walk-forward 기준으로 cross-attention 재실험 — 단, KV에 넣을 그룹이 개별로 ≥0.55는 돼야 의미 있음. 현재는 조건 미충족.
3. T1 label 재정의 검토(horizon chunk 출력, 다중 horizon head) — 과제 평가와 호환되는지 먼저 확인.
4. 외부 데이터 없이 T1을 더 올릴 여지는 작음. 보고서에는 진단(manifold·미분)을 근거로 "예측력 약함"을 명시하는 쪽이 설득력 있음.
5. H200 사용 가능 시: TabPFN T1 전문가를 CPU fallback 없이 재생성, seed 늘려 se 축소.
