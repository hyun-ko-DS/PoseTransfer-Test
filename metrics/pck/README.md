# PCK/PCKh 평가 모듈

## 개요

PCK (Percentage of Correct Keypoints) 및 PCKh (PCK with head size normalization) 메트릭을 계산하여 포즈 전송의 정확도를 평가합니다.

## 설치

OpenPose를 사용하기 위해서는 OpenPose를 설치하고 Python 바인딩을 설정해야 합니다.

### 방법 1: pyopenpose 사용 (권장)

```bash
# OpenPose를 먼저 설치한 후
pip install opencv-python numpy
# pyopenpose는 OpenPose 빌드에 포함되어 있습니다
```

### 방법 2: OpenPose 직접 설치

OpenPose 공식 문서를 참고하여 설치:
https://github.com/CMU-Perceptual-Computing-Lab/openpose

설치 후 Python 바인딩을 사용할 수 있습니다.

## 사용 방법

### 기본 사용

```python
from metrics.pck import PCKEvaluator
from pathlib import Path

# 평가기 초기화
evaluator = PCKEvaluator()

# 생성된 이미지와 참조 키포인트 이미지 비교
result = evaluator.evaluate(
    generated_image_path="results/nano_banana/short/full/bg/full_bg_1_kp_1.jpg",
    reference_keypoint_path="data/keypoints/full/kp_full_1.png",
    image_type="full"
)

print(result)
# {
#     "pck_0.1": 0.85,
#     "pck_0.2": 0.92,
#     "pckh_0.5": 0.88,
#     "mean_keypoint_error": 12.5
# }
```

### 일괄 평가

```python
from metrics.pck import evaluate_pck_batch

generated_images = [
    "results/nano_banana/short/full/bg/full_bg_1_kp_1.jpg",
    "results/nano_banana/short/full/bg/full_bg_1_kp_2.jpg",
]

reference_keypoints = [
    "data/keypoints/full/kp_full_1.png",
    "data/keypoints/full/kp_full_2.png",
]

image_types = ["full", "full"]

results = evaluate_pck_batch(
    generated_images=generated_images,
    reference_keypoints=reference_keypoints,
    image_types=image_types,
    output_path="evaluations/pck_results.json"
)
```

## 메트릭 설명

### PCK (Percentage of Correct Keypoints)

- **정의**: 예측 키포인트가 정답 키포인트로부터 특정 거리(이미지 대각선의 α 배) 내에 있는 비율
- **계산**: `PCK@α = (올바른 키포인트 수) / (전체 키포인트 수)`
- **임계값**: 일반적으로 0.1, 0.2 사용

### PCKh (PCK with head size normalization)

- **정의**: PCK와 동일하지만, 거리 임계값을 머리 크기로 정규화
- **계산**: `PCKh@α = (올바른 키포인트 수) / (전체 키포인트 수)`
- **임계값**: 일반적으로 0.5 사용 (머리 크기의 50%)

### Mean Keypoint Error

- **정의**: 모든 키포인트의 평균 유클리드 거리 오차
- **단위**: 픽셀

## 키포인트 추출

키포인트 추출은 OpenPose를 사용합니다.

```python
from extractor.openpose import KeypointExtractor

extractor = KeypointExtractor()

# 이미지에서 키포인트 추출
keypoints = extractor.extract_from_image(
    image_path="path/to/image.jpg",
    image_type="full"  # 또는 "half"
)

# 키포인트 딕셔너리: {키포인트명: (x, y, visibility), ...}
print(keypoints)
```

## 주의사항

1. **키포인트 이미지 처리**: 입력 키포인트 이미지는 스켈레톤 이미지이므로, MediaPipe로 추출하면 스켈레톤 이미지에서 사람을 찾으려고 시도합니다. 더 정확한 평가를 위해서는 키포인트 좌표를 직접 파싱하는 방법을 고려할 수 있습니다.

2. **이미지 타입**: `image_type`은 'full' 또는 'half'를 사용합니다. 'half'인 경우 상반신 키포인트만 사용됩니다.

3. **Confidence**: OpenPose의 confidence가 0.3 미만인 키포인트는 평가에서 제외됩니다.

## OpenPose 설정

KeypointExtractor를 초기화할 때 OpenPose 경로를 지정할 수 있습니다:

```python
extractor = KeypointExtractor(
    openpose_path="/path/to/openpose",
    model_folder="/path/to/openpose/models"
)
```

경로를 지정하지 않으면 환경 변수나 기본 경로를 사용합니다.

