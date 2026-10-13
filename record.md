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

---

## 2026-10-13 · 세션 2 (lead feature 발견 이후)

### 전환점: causal 가정이 틀렸다
- 지금까지 모든 feature를 과거 방향으로만 만들었음. 과제는 배치 채점이라 test 파일에 925~1124일 feature가 전부 주어짐 → t일 label(KOSPI t→t+3)을 맞힐 때 **t+1~t+3일 feature(lead)** 사용 가능.
- 단일 feature: USD/KRW t+3−t 변화만으로 AUC 0.739(역방향), S&P500 (t+2)−(t−1) 0.715 (미국 지수는 하루 시차 정렬이 최적). KE/SH/WE lead는 신호 없음.
- 리더보드 상위 T1 0.92는 lead 사용으로 추정. 우리 검증 상한은 0.81.

### 규정 검토 (docs PDF 25쪽)
- ① 외부 데이터 사용 금지 ② 평가용 데이터는 학습에 활용 금지. "TTA 금지" 문구는 없음.
- lead는 ②에 대해: 학습 행 922~924의 lead가 test 행을 참조 → **학습에서 제외**(`MAX_LEAD=3`). 예측 시 test 행 간 참조는 학습이 아니라 입력이므로 문구상 허용. 13쪽 "예측 시점에 알 수 없는 정보" 정의에는 걸릴 수 있음 → 보고서에 명시.
- ①에 대해: TabPFN·EXAONE 같은 사전학습 모델은 문구상 "데이터"는 아니나 해석 리스크 + 재현성(토큰·라이선스) 리스크 → **최종 파이프라인에서 제외**, 실험 비교에만 사용. KD(teacher=TabPFN)도 같은 이유로 비상용.

### 측정 결과 — T1 (lead feature, walk-forward)
| 구성 | wf5 | wf10 | time20 | fold 최저(wf5) | s100 min/med/max |
|---|---|---|---|---|---|
| TabPFN fi_k3+z (zscore + FI t+3) | 0.805 | 0.808 | 0.768 | 0.655 | 0.64/0.81/0.88 |
| XGB fi_al (FI 정렬 lead 90 + 요일, base 없음) | 0.794 | 0.801 | 0.777 | 0.672 | 0.67/0.80/0.88 |
| ResMLP / ResConv1D / xattn_nocross (lead 입력) | 0.79 | 0.79 | – | 0.68~0.70 | 0.66~0.68/0.76~0.80/0.85~0.88 |
| LeadXAttn (cross-attn, gate 0.02) | 0.788 | – | – | 0.679 | – |
| KD (teacher TabPFN) 4종 | 0.763~0.788 | – | – | – | 전 구조에서 비-KD보다 낮음 |
| **conv1d + XGB fi_al 순위평균 (최종)** | **0.804** | **0.806** | – | **0.695** | 0.68/0.80/0.88 |
- 예측 평활(인접일 평균): 효과 없음(±0.006). 신경망 train AUC 0.86~0.87 vs val 0.79 (과적합 폭 0.07).
- 신경망 4종 OOF 상관 0.88~0.94 → 사실상 동일 모델. cross-attn은 lead 입력에서도 KV를 쓰지 않음.

### 측정 결과 — T2
- XGB level+change + 전체 lead(k=1..3): wf5 0.982 / wf10 0.982 / time20 0.971, 5 fold 모두 causal TabPFN(0.972)보다 높음. 확정.

### 제출 이력
- public `_3`(causal TabPFN), `_5`(TabPFN fi_k123, 922~924 포함), `_7`(TabPFN fi_k3, 규정 준수), `_8`(XGB fi_al). `_6`은 `_7`과 동일(사용 안 함).
- 리더보드: `_7`·`_8` "약간 높거나 비슷" → 검증 예상(0.75~0.85)과 일치.
- private 최종: `final_nokd.ipynb` 산출(T1 conv1d+XGB 블렌드, T2 XGB) 예정. causal 버전(`private_3반_안중범_tabpfn_causal.csv`)은 보관.

### 회고
- 가장 큰 교훈: **과제의 채점 형식을 먼저 확인할 것.** "예측"이라는 단어에 묶여 하루 종일 causal 천장(0.59)과 싸움. 리더보드 수치를 보고서야 발견.
- 신경망은 lead 입력에서 즉시 0.79에 도달 → 모델이 아니라 정보가 문제였다는 세션 1 결론을 재확인.
- 선택 규칙을 미리 정해 둔 것(wf5·wf10·fold 최저 일관성)이 노이즈 속 과적합 선택을 막음.
- 삽질: KD teacher OOF, 전문가 OOF 등 TabPFN 기반 산출물은 결국 규정 리스크로 폐기. 규정 검토를 먼저 했으면 1~2시간 절약.

### 다음
1. `final_nokd.ipynb` 실행·검증 → private 파일 확정, git 커밋.
2. report.html에 세션 2(lead, 규정, 신경망, 블렌드) 추가. 보고서 PDF.
3. 남으면: 블렌드 가중치·W·dropout 민감도(wf10 기준) 소규모 확인.
