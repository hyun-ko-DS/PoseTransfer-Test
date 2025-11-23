# LPIPS (Learned Perceptual Image Patch Similarity) 메트릭

## 개요

LPIPS는 두 이미지 간의 시각적 유사도를 측정하는 딥러닝 기반 메트릭입니다. 
픽셀 단위 차이가 아닌, 인간이 인지하는 시각적 차이를 측정합니다.

**점수 범위**: 0 (완전히 유사) ~ 1 (완전히 다름)
- **낮을수록 좋음**: 0에 가까울수록 원본과 생성 이미지가 시각적으로 유사함

## 설치

```bash
pip install lpips torch torchvision
```

## 사용법

### 기본 사용

```python
from metrics.lpips import calculate_lpips

# 이미지 경로로 계산
score = calculate_lpips('original.jpg', 'generated.jpg')
print(f"LPIPS score: {score:.4f}")

# PIL Image 객체로 계산
from PIL import Image
img0 = Image.open('original.jpg')
img1 = Image.open('generated.jpg')
score = calculate_lpips(img0, img1)
```

### 일괄 평가

```python
from metrics.lpips import evaluate_lpips_batch

original_images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
generated_images = ['gen1.jpg', 'gen2.jpg', 'gen3.jpg']

results = evaluate_lpips_batch(
    original_images,
    generated_images,
    output_path='lpips_results.json'
)
```

### 네트워크 선택

LPIPS는 여러 사전 학습된 네트워크를 지원합니다:

```python
# AlexNet 기반 (기본값, 빠름)
score = calculate_lpips('img1.jpg', 'img2.jpg', net='alex')

# VGG 기반 (더 정확하지만 느림)
score = calculate_lpips('img1.jpg', 'img2.jpg', net='vgg')

# SqueezeNet 기반 (가장 빠름)
score = calculate_lpips('img1.jpg', 'img2.jpg', net='squeeze')
```

## 결과 형식

### 단일 평가 결과

```python
score = calculate_lpips('original.jpg', 'generated.jpg')
# 반환값: float (예: 0.234)
```

### 일괄 평가 결과

```json
{
  "statistics": {
    "mean": 0.234,
    "std": 0.045,
    "min": 0.156,
    "max": 0.389,
    "total_images": 75,
    "valid_images": 75
  },
  "results": [
    {
      "original_image": "data/images/full/bg/full_bg_1.jpg",
      "generated_image": "results/nano_banana/short/full/bg/full_bg_1_kp_1.png",
      "lpips_score": 0.234,
      "index": 0
    },
    ...
  ]
}
```

## 우리 태스크에서의 활용

### 평가 목적
- **원본 이미지와 생성 이미지 간의 시각적 유사도 측정**
- 포즈는 다르지만 얼굴, 옷, 배경 등이 얼마나 유지되었는지 평가

### 해석 가이드
- **0.0 ~ 0.2**: 매우 유사 (거의 동일한 시각적 품질)
- **0.2 ~ 0.4**: 유사 (일부 차이 있지만 전반적으로 유사)
- **0.4 ~ 0.6**: 보통 (눈에 띄는 차이 있음)
- **0.6 ~ 0.8**: 다름 (상당한 차이)
- **0.8 ~ 1.0**: 매우 다름 (거의 다른 이미지)

### 주의사항
- LPIPS는 **포즈 차이를 고려하지 않습니다**
- 포즈가 다르더라도 얼굴, 옷 등이 유지되면 낮은 점수(유사)를 받습니다
- 포즈 정확도는 **PCK/PCKh 메트릭**으로 별도 평가해야 합니다

## 성능

- **AlexNet**: 빠르고 효율적 (기본값 권장)
- **VGG**: 더 정확하지만 느림
- **SqueezeNet**: 가장 빠르지만 정확도는 낮음

GPU 사용 시 훨씬 빠르게 계산됩니다.

## 참고 자료

- 원본 논문: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (Zhang et al., 2018)
- GitHub: https://github.com/richzhang/PerceptualSimilarity

