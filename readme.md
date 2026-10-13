#### 데이터분석 접근방향

1. 전처리 및 서브셋 결정
2. 모델 탐색 및 파생변수 설정
3. 오버피팅 방지 기법
4. test-time compute기법
5. 하이퍼파라미터 튜닝

#### 모델후보
- TabPFN 3.5(base)
- Exaone forecast
- Exaone finance(T1)
- GBM(base: xgboost)
- 1D Conv+FFN

### 문제인식
T2는 AUC가 높지만, T1은 매우낮으며, 최종 평가는 평균을 계산하므로 T1의 성능을 올리는것이 관건


#현재상황
1. T2 확정
2. 탐색중 TabPFN을 넘기기 쉽지않음.
