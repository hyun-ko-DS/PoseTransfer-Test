"""
PCK/PCKh (Percentage of Correct Keypoints) 평가 모듈

입력 키포인트와 생성된 이미지에서 추출한 키포인트를 비교하여
포즈 유사도를 측정합니다.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from extractor.openpose import KeypointExtractor


class PCKEvaluator:
    """
    PCK/PCKh 평가를 수행하는 클래스
    """
    
    def __init__(self, openpose_path: Optional[str] = None, model_folder: Optional[str] = None):
        """
        PCK 평가기 초기화
        
        Args:
            openpose_path: OpenPose 빌드 경로 (선택사항)
            model_folder: OpenPose 모델 폴더 경로 (선택사항)
        """
        self.extractor = KeypointExtractor(openpose_path=openpose_path, model_folder=model_folder)
    
    def calculate_pck(
        self,
        predicted_keypoints: Dict[str, Tuple[float, float, float]],
        ground_truth_keypoints: Dict[str, Tuple[float, float, float]],
        threshold: float = 0.2,
        image_size: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        PCK (Percentage of Correct Keypoints)를 계산합니다.
        
        Args:
            predicted_keypoints: 예측된 키포인트 딕셔너리
            ground_truth_keypoints: 정답 키포인트 딕셔너리
            threshold: 임계값 (이미지 대각선 길이의 비율)
            image_size: 이미지 크기 (width, height), None이면 자동 계산
        
        Returns:
            PCK 점수 (0.0 ~ 1.0)
        """
        if image_size is None:
            # 키포인트 좌표로부터 이미지 크기 추정
            all_coords = []
            for kp_dict in [predicted_keypoints, ground_truth_keypoints]:
                for x, y, _ in kp_dict.values():
                    all_coords.extend([x, y])
            if all_coords:
                max_coord = max(all_coords)
                # 대략적인 이미지 크기 추정
                image_size = (int(max_coord) + 100, int(max_coord) + 100)
            else:
                image_size = (512, 512)
        
        # 이미지 대각선 길이
        diagonal = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
        threshold_pixels = threshold * diagonal
        
        correct_count = 0
        total_count = 0
        
        # 공통 키포인트에 대해 비교
        common_keys = set(predicted_keypoints.keys()) & set(ground_truth_keypoints.keys())
        
        for key in common_keys:
            pred_x, pred_y, pred_vis = predicted_keypoints[key]
            gt_x, gt_y, gt_vis = ground_truth_keypoints[key]
            
            # visibility가 너무 낮으면 제외
            if pred_vis < 0.3 or gt_vis < 0.3:
                continue
            
            # 유클리드 거리 계산
            distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            
            if distance <= threshold_pixels:
                correct_count += 1
            
            total_count += 1
        
        if total_count == 0:
            return 0.0
        
        return correct_count / total_count
    
    def calculate_pckh(
        self,
        predicted_keypoints: Dict[str, Tuple[float, float, float]],
        ground_truth_keypoints: Dict[str, Tuple[float, float, float]],
        threshold: float = 0.5
    ) -> float:
        """
        PCKh (PCK with head size normalization)를 계산합니다.
        
        Args:
            predicted_keypoints: 예측된 키포인트 딕셔너리
            ground_truth_keypoints: 정답 키포인트 딕셔너리
            threshold: 임계값 (머리 크기의 비율)
        
        Returns:
            PCKh 점수 (0.0 ~ 1.0)
        """
        # 정답 키포인트에서 머리 크기 계산
        head_size = self.extractor.calculate_head_size(ground_truth_keypoints)
        
        if head_size <= 0:
            return 0.0
        
        threshold_pixels = threshold * head_size
        
        correct_count = 0
        total_count = 0
        
        # 공통 키포인트에 대해 비교
        common_keys = set(predicted_keypoints.keys()) & set(ground_truth_keypoints.keys())
        
        for key in common_keys:
            pred_x, pred_y, pred_vis = predicted_keypoints[key]
            gt_x, gt_y, gt_vis = ground_truth_keypoints[key]
            
            # visibility가 너무 낮으면 제외
            if pred_vis < 0.3 or gt_vis < 0.3:
                continue
            
            # 유클리드 거리 계산
            distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            
            if distance <= threshold_pixels:
                correct_count += 1
            
            total_count += 1
        
        if total_count == 0:
            return 0.0
        
        return correct_count / total_count
    
    def calculate_mean_keypoint_error(
        self,
        predicted_keypoints: Dict[str, Tuple[float, float, float]],
        ground_truth_keypoints: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """
        평균 키포인트 오차를 계산합니다.
        
        Args:
            predicted_keypoints: 예측된 키포인트 딕셔너리
            ground_truth_keypoints: 정답 키포인트 딕셔너리
        
        Returns:
            평균 오차 (픽셀 단위)
        """
        errors = []
        
        # 공통 키포인트에 대해 비교
        common_keys = set(predicted_keypoints.keys()) & set(ground_truth_keypoints.keys())
        
        for key in common_keys:
            pred_x, pred_y, pred_vis = predicted_keypoints[key]
            gt_x, gt_y, gt_vis = ground_truth_keypoints[key]
            
            # visibility가 너무 낮으면 제외
            if pred_vis < 0.3 or gt_vis < 0.3:
                continue
            
            # 유클리드 거리 계산
            distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            errors.append(distance)
        
        if len(errors) == 0:
            return 0.0
        
        return np.mean(errors)
    
    def evaluate(
        self,
        generated_image_path: Union[str, Path],
        reference_keypoint_path: Union[str, Path],
        image_type: str = "full"
    ) -> Dict[str, float]:
        """
        생성된 이미지와 참조 키포인트를 비교하여 PCK/PCKh를 계산합니다.
        
        Args:
            generated_image_path: 생성된 이미지 경로
            reference_keypoint_path: 참조 키포인트 이미지 경로
            image_type: 'full' 또는 'half'
        
        Returns:
            평가 결과 딕셔너리
        """
        # 생성된 이미지에서 키포인트 추출
        pred_keypoints = self.extractor.extract_from_image(
            generated_image_path, 
            image_type=image_type
        )
        
        if pred_keypoints is None:
            return {
                "pck_0.1": 0.0,
                "pck_0.2": 0.0,
                "pckh_0.5": 0.0,
                "mean_keypoint_error": 0.0,
                "error": "No person detected in generated image"
            }
        
        # 참조 키포인트 이미지에서 키포인트 추출
        ref_keypoints = self.extractor.extract_from_image(
            reference_keypoint_path,
            image_type=image_type
        )
        
        if ref_keypoints is None:
            return {
                "pck_0.1": 0.0,
                "pck_0.2": 0.0,
                "pckh_0.5": 0.0,
                "mean_keypoint_error": 0.0,
                "error": "No person detected in reference keypoint image"
            }
        
        # PCK 계산
        pck_01 = self.calculate_pck(pred_keypoints, ref_keypoints, threshold=0.1)
        pck_02 = self.calculate_pck(pred_keypoints, ref_keypoints, threshold=0.2)
        
        # PCKh 계산
        pckh_05 = self.calculate_pckh(pred_keypoints, ref_keypoints, threshold=0.5)
        
        # 평균 오차 계산
        mean_error = self.calculate_mean_keypoint_error(pred_keypoints, ref_keypoints)
        
        return {
            "pck_0.1": float(pck_01),
            "pck_0.2": float(pck_02),
            "pckh_0.5": float(pckh_05),
            "mean_keypoint_error": float(mean_error)
        }


def calculate_pck(
    predicted_keypoints: Dict[str, Tuple[float, float, float]],
    ground_truth_keypoints: Dict[str, Tuple[float, float, float]],
    threshold: float = 0.2
) -> float:
    """
    편의 함수: PCK를 계산합니다.
    """
    evaluator = PCKEvaluator()
    return evaluator.calculate_pck(predicted_keypoints, ground_truth_keypoints, threshold)


def calculate_pckh(
    predicted_keypoints: Dict[str, Tuple[float, float, float]],
    ground_truth_keypoints: Dict[str, Tuple[float, float, float]],
    threshold: float = 0.5
) -> float:
    """
    편의 함수: PCKh를 계산합니다.
    """
    evaluator = PCKEvaluator()
    return evaluator.calculate_pckh(predicted_keypoints, ground_truth_keypoints, threshold)


def evaluate_pck_batch(
    generated_images: List[Union[str, Path]],
    reference_keypoints: List[Union[str, Path]],
    image_types: List[str],
    output_path: Optional[Union[str, Path]] = None
) -> List[Dict]:
    """
    여러 이미지에 대해 일괄 PCK 평가를 수행합니다.
    
    Args:
        generated_images: 생성된 이미지 경로 리스트
        reference_keypoints: 참조 키포인트 이미지 경로 리스트
        image_types: 이미지 타입 리스트 ('full' 또는 'half')
        output_path: 결과를 저장할 JSON 파일 경로 (선택사항)
    
    Returns:
        평가 결과 리스트
    """
    evaluator = PCKEvaluator()
    results = []
    
    for gen_img, ref_kp, img_type in zip(generated_images, reference_keypoints, image_types):
        result = evaluator.evaluate(gen_img, ref_kp, img_type)
        result["generated_image"] = str(gen_img)
        result["reference_keypoint"] = str(ref_kp)
        result["image_type"] = img_type
        results.append(result)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ PCK 평가 결과 저장: {output_path}")
    
    return results

