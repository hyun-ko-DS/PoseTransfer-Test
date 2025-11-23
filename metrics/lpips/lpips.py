"""
LPIPS (Learned Perceptual Image Patch Similarity) 메트릭 구현

원본 이미지와 생성된 이미지 간의 시각적 유사도를 측정합니다.
0에 가까울수록 유사하고, 1에 가까울수록 다릅니다.
"""

import os
import json
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  lpips 라이브러리가 설치되지 않았습니다. 'pip install lpips'를 실행하세요.")


class LPIPSEvaluator:
    """LPIPS 메트릭 계산 클래스"""
    
    def __init__(self, net: str = 'vgg', device: Optional[str] = None):
        """
        LPIPS 평가기 초기화
        
        Args:
            net: 사용할 네트워크 ('alex', 'vgg', 'squeeze')
            device: 사용할 디바이스 ('cuda', 'cpu', None=자동)
        """
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips 라이브러리가 필요합니다. 'pip install lpips'를 실행하세요."
            )
        
        # 디바이스 설정 (우선순위: cuda > mps > cpu)
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        # LPIPS 모델 로드
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()  # 평가 모드
        
        print(f"✓ LPIPS 평가기 초기화 완료 (네트워크: {net}, 디바이스: {device})")
    
    def _preprocess_image(self, img: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        PIL Image를 LPIPS 입력 형식으로 변환
        
        Args:
            img: PIL Image 객체
            target_size: 리사이즈할 크기 (width, height), None이면 원본 크기 유지
        
        Returns:
            torch.Tensor: (1, 3, H, W) 형태의 텐서, 값 범위 [-1, 1]
        """
        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 크기 조정 (필요한 경우)
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        
        # numpy 배열로 변환
        img_np = np.array(img).astype(np.float32)
        
        # [0, 255] -> [0, 1] -> [-1, 1]로 정규화
        img_np = img_np / 255.0
        img_np = (img_np - 0.5) / 0.5
        
        # (H, W, C) -> (C, H, W)로 변환
        img_np = img_np.transpose(2, 0, 1)
        
        # 배치 차원 추가: (C, H, W) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def calculate(self, img0: Image.Image, img1: Image.Image) -> float:
        """
        두 이미지 간의 LPIPS 점수 계산
        
        Args:
            img0: 원본 이미지 (PIL Image)
            img1: 생성된 이미지 (PIL Image)
        
        Returns:
            float: LPIPS 점수 (0에 가까울수록 유사, 1에 가까울수록 다름)
        """
        # 두 이미지의 크기를 맞춤 (더 작은 크기로 리사이즈)
        size0 = img0.size  # (width, height)
        size1 = img1.size
        
        # 두 이미지 중 더 작은 크기로 통일
        target_size = (min(size0[0], size1[0]), min(size0[1], size1[1]))
        
        # 이미지 전처리 (크기 맞춤)
        img0_tensor = self._preprocess_image(img0, target_size=target_size)
        img1_tensor = self._preprocess_image(img1, target_size=target_size)
        
        # LPIPS 계산
        with torch.no_grad():
            distance = self.loss_fn(img0_tensor, img1_tensor)
        
        # 스칼라 값으로 변환
        return distance.item()
    
    def calculate_from_paths(
        self, 
        img0_path: Union[str, Path], 
        img1_path: Union[str, Path]
    ) -> float:
        """
        파일 경로로부터 두 이미지 간의 LPIPS 점수 계산
        
        Args:
            img0_path: 원본 이미지 경로
            img1_path: 생성된 이미지 경로
        
        Returns:
            float: LPIPS 점수
        """
        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        return self.calculate(img0, img1)


# 전역 평가기 인스턴스 (lazy initialization)
_evaluator: Optional[LPIPSEvaluator] = None


def get_evaluator(net: str = 'vgg', device: Optional[str] = None) -> LPIPSEvaluator:
    """LPIPS 평가기 싱글톤 인스턴스 반환"""
    global _evaluator
    
    # 디바이스 우선순위 설정 (cuda > mps > cpu)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    if _evaluator is None:
        _evaluator = LPIPSEvaluator(net=net, device=device)
    return _evaluator


def calculate_lpips(
    original_image: Union[str, Path, Image.Image],
    generated_image: Union[str, Path, Image.Image],
    net: str = 'vgg',
    device: Optional[str] = None
) -> float:
    """
    두 이미지 간의 LPIPS 점수를 계산합니다.
    
    Args:
        original_image: 원본 이미지 (경로 또는 PIL Image)
        generated_image: 생성된 이미지 (경로 또는 PIL Image)
        net: 사용할 네트워크 ('alex', 'vgg', 'squeeze')
        device: 사용할 디바이스 ('cuda', 'cpu', None=자동)
    
    Returns:
        float: LPIPS 점수 (0에 가까울수록 유사, 1에 가까울수록 다름)
    
    Example:
        >>> score = calculate_lpips('original.jpg', 'generated.jpg')
        >>> print(f"LPIPS score: {score:.4f}")
    """
    evaluator = get_evaluator(net=net, device=device)
    
    # 이미지 로드
    if isinstance(original_image, (str, Path)):
        img0 = Image.open(original_image)
    else:
        img0 = original_image
    
    if isinstance(generated_image, (str, Path)):
        img1 = Image.open(generated_image)
    else:
        img1 = generated_image
    
    return evaluator.calculate(img0, img1)


def evaluate_lpips_batch(
    original_images: List[Union[str, Path]],
    generated_images: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    net: str = 'alex',
    device: Optional[str] = None
) -> List[Dict]:
    """
    여러 이미지 쌍에 대해 LPIPS 점수를 일괄 계산합니다.
    
    Args:
        original_images: 원본 이미지 경로 리스트
        generated_images: 생성된 이미지 경로 리스트
        output_path: 결과를 저장할 JSON 파일 경로 (선택사항)
        net: 사용할 네트워크 ('alex', 'vgg', 'squeeze')
        device: 사용할 디바이스 ('cuda', 'cpu', None=자동)
    
    Returns:
        List[Dict]: 각 이미지 쌍의 LPIPS 점수와 메타데이터 리스트
    
    Example:
        >>> results = evaluate_lpips_batch(
        ...     ['img1.jpg', 'img2.jpg'],
        ...     ['gen1.jpg', 'gen2.jpg'],
        ...     output_path='lpips_results.json'
        ... )
    """
    if len(original_images) != len(generated_images):
        raise ValueError(
            f"원본 이미지 수({len(original_images)})와 "
            f"생성 이미지 수({len(generated_images)})가 일치하지 않습니다."
        )
    
    evaluator = get_evaluator(net=net, device=device)
    results = []
    
    print(f"LPIPS 평가 시작: {len(original_images)}개 이미지 쌍")
    
    for i, (orig_path, gen_path) in enumerate(zip(original_images, generated_images)):
        try:
            # LPIPS 점수 계산
            score = evaluator.calculate_from_paths(orig_path, gen_path)
            
            result = {
                "original_image": str(orig_path),
                "generated_image": str(gen_path),
                "lpips_score": float(score),
                "index": i
            }
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  진행: {i + 1}/{len(original_images)}")
        
        except Exception as e:
            print(f"⚠️  오류 발생 ({orig_path} vs {gen_path}): {str(e)}")
            results.append({
                "original_image": str(orig_path),
                "generated_image": str(gen_path),
                "lpips_score": None,
                "error": str(e),
                "index": i
            })
    
    # 통계 계산
    valid_scores = [r["lpips_score"] for r in results if r["lpips_score"] is not None]
    if valid_scores:
        stats = {
            "mean": float(np.mean(valid_scores)),
            "std": float(np.std(valid_scores)),
            "min": float(np.min(valid_scores)),
            "max": float(np.max(valid_scores)),
            "total_images": len(original_images),
            "valid_images": len(valid_scores)
        }
    else:
        stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "total_images": len(original_images),
            "valid_images": 0
        }
    
    output = {
        "statistics": stats,
        "results": results
    }
    
    # 결과 저장
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✓ 결과 저장: {output_path}")
    
    # 통계 출력
    if stats["mean"] is not None:
        print(f"\nLPIPS 평가 완료:")
        print(f"  평균: {stats['mean']:.4f}")
        print(f"  표준편차: {stats['std']:.4f}")
        print(f"  최소값: {stats['min']:.4f}")
        print(f"  최대값: {stats['max']:.4f}")
    
    return results

