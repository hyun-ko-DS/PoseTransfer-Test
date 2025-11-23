# 평가 지표 계획

## 1. 원본과 생성 이미지 간의 시각적 유사도

### 메트릭: LPIPS (Learned Perceptual Image Patch Similarity)

**목적**: 포즈는 다르지만 얼굴, 옷, 배경 등이 얼마나 유사한지 측정

**구현 방법**:
- `lpips` 라이브러리 사용
- VGG 또는 AlexNet 기반 모델 선택
- 값 범위: 0 (완전히 유사) ~ 1 (완전히 다름)
- 낮을수록 좋음

**저장 형식**:
```json
{
  "image_name": "full_bg_1_kp_1",
  "lpips_score": 0.234,
  "model": "nano_banana",
  "prompt": "short"
}
```

---

## 2. 키포인트와 생성 이미지의 포즈 간의 유사도

### 메트릭: PCK / PCKh (Percentage of Correct Keypoints)

**목적**: 입력 키포인트가 생성 이미지에 얼마나 정확히 반영되었는지 측정

**구현 방법**:
1. 생성된 이미지에서 키포인트 추출 (OpenPose, MediaPipe 등)
2. 입력 키포인트와 추출된 키포인트 비교
3. PCK@0.1, PCK@0.2 등 여러 임계값으로 계산
4. PCKh는 머리 크기로 정규화

**계산 공식**:
```
PCK@α = (올바른 키포인트 수) / (전체 키포인트 수)
- 올바른 키포인트: 예측 위치가 정답으로부터 α × head_size 거리 내에 있는 경우
```

**저장 형식**:
```json
{
  "image_name": "full_bg_1_kp_1",
  "pck_0.1": 0.85,
  "pck_0.2": 0.92,
  "pckh_0.5": 0.88,
  "mean_keypoint_error": 12.5,
  "model": "nano_banana",
  "prompt": "short"
}
```

---

## 3. 비용 및 시간 관련 메트릭

### 메트릭: 토큰 수 및 추론 시간

**목적**: 모델의 효율성과 비용 측정

**측정 항목**:
- **입력 토큰 수**: 원본 이미지 + 키포인트 이미지 + 프롬프트
- **출력 토큰 수**: 생성된 이미지 (또는 API 응답 크기)
- **추론 시간**: API 호출 시작부터 완료까지의 시간
- **비용 추정**: 토큰 수 기반 비용 계산 (API별 요금제에 따라)

**저장 형식**:
```json
{
  "image_name": "full_bg_1_kp_1",
  "input_tokens": 1500,
  "output_tokens": 800,
  "total_tokens": 2300,
  "inference_time_seconds": 3.45,
  "estimated_cost": 0.0023,
  "model": "nano_banana",
  "prompt": "short"
}
```

**구현 방법**:
- `elapse_time()` 함수 활용 (이미 구현됨)
- API 응답에서 토큰 정보 추출 (가능한 경우)
- 각 모델별 비용 계산 함수 구현

---

## 4. 정성적 평가 방법

### 사람이 눈으로 평가

**목적**: 자동화된 메트릭으로 측정하기 어려운 품질 요소 평가

**평가 항목**:
1. **포즈 정확도**: 키포인트가 얼마나 정확히 반영되었는가? (1-5점)
2. **얼굴 유사도**: 원본 얼굴이 얼마나 유지되었는가? (1-5점)
3. **자연스러움**: 생성된 이미지가 얼마나 자연스러운가? (1-5점)
4. **전체 품질**: 종합적인 품질 평가 (1-5점)

**평가 시트 형식**:
```csv
image_name,model,prompt,evaluator,pose_accuracy,face_similarity,naturalness,overall_quality,comments
full_bg_1_kp_1,nano_banana,short,evaluator_1,4,5,4,4,"좋은 결과"
full_bg_1_kp_1,nano_banana,short,evaluator_2,3,4,5,4,"자연스러움 우수"
```

**구현 방법**:
- 웹 인터페이스 또는 스프레드시트로 평가
- 여러 평가자의 평균 점수 계산
- 평가자 간 일치도 (Inter-rater agreement) 계산

---

## 종합 평가 결과 저장 구조

### 디렉토리 구조
```
results/
├── nano_banana/
│   ├── short/
│   │   ├── full/
│   │   │   ├── bg/
│   │   │   │   ├── full_bg_1_kp_1.png
│   │   │   │   └── ...
│   │   │   └── no_bg/
│   │   └── ...
│   └── ...
└── ...

evaluations/
├── quantitative/
│   ├── lpips_results.json
│   ├── pck_results.json
│   └── cost_time_results.json
└── qualitative/
    └── human_evaluations.csv
```

### 종합 리포트 형식
```json
{
  "model": "nano_banana",
  "prompt": "short",
  "total_images": 75,
  "metrics": {
    "lpips": {
      "mean": 0.234,
      "std": 0.045,
      "min": 0.156,
      "max": 0.389
    },
    "pck": {
      "pck_0.1": 0.85,
      "pck_0.2": 0.92,
      "mean_keypoint_error": 12.5
    },
    "cost_time": {
      "mean_inference_time": 3.45,
      "total_tokens": 172500,
      "estimated_total_cost": 0.1725
    },
    "qualitative": {
      "mean_pose_accuracy": 4.2,
      "mean_face_similarity": 4.5,
      "mean_naturalness": 4.3,
      "mean_overall_quality": 4.3
    }
  }
}
```

---

## 구현 우선순위

1. **Phase 1**: 비용/시간 메트릭 (이미 부분적으로 구현됨)
2. **Phase 2**: LPIPS 메트릭 구현
3. **Phase 3**: PCK/PCKh 메트릭 구현 (키포인트 추출 필요)
4. **Phase 4**: 정성적 평가 시스템 구축

---

## 추가 고려사항

### 통계 분석
- 모델별, 프롬프트별 성능 비교
- 이미지 타입별 (full/half/selfie) 성능 차이
- 키포인트 번호별 (kp_1, kp_2, kp_3) 성능 차이

### 시각화
- 메트릭별 히스토그램
- 모델 비교 바 차트
- 산점도 (LPIPS vs PCK 등)

### 자동화
- 이미지 생성 후 자동으로 정량적 메트릭 계산
- 결과를 JSON/CSV로 자동 저장
- 리포트 자동 생성

