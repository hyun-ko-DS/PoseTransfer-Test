"""
extracted_keypoints의 original과 생성 모델 간 PCK/PCKh 평가 스크립트

extracted_keypoints/original/{image_type}/의 JSON 파일과
extracted_keypoints/{model_name}/{prompt_version}/{image_type}/의 JSON 파일을 비교하여
PCK/PCKh 점수를 계산합니다.

파일명 매칭 규칙:
- original/full/kp_full_1_origin_keypoints.json ↔ nano_banana/short/full/full_bg_1_kp_1_keypoints.json
- kp_ 뒤의 숫자로 매칭 (bg, no_bg는 무시)
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from PIL import Image

# OpenPose BODY_25 키포인트 이름 (25개)
KEYPOINT_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel"
]


def load_keypoints_from_json(json_path: Path) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """
    OpenPose JSON 파일에서 키포인트를 로드합니다.
    
    Args:
        json_path: OpenPose JSON 파일 경로
    
    Returns:
        키포인트 딕셔너리: {키포인트명: (x, y, confidence), ...} 또는 None
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'people' not in data or len(data['people']) == 0:
            return None
        
        # 첫 번째 사람의 키포인트만 사용
        person = data['people'][0]
        
        # BODY_25 키포인트 (25개, 각각 [x, y, confidence])
        pose_keypoints = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
        
        if len(pose_keypoints) == 0:
            return None
        
        # 키포인트 딕셔너리로 변환
        keypoints = {}
        
        for idx, name in enumerate(KEYPOINT_NAMES):
            if idx < len(pose_keypoints):
                x, y, confidence = pose_keypoints[idx]
                if confidence > 0:
                    keypoints[name] = (float(x), float(y), float(confidence))
        
        return keypoints if keypoints else None
    except Exception as e:
        print(f"⚠️  JSON 로드 실패 ({json_path}): {e}")
        return None


def extract_kp_num_from_original_json(filename: str) -> Optional[str]:
    """
    original 폴더의 JSON 파일명에서 키포인트 번호를 추출합니다.
    
    예: "kp_full_1_origin_keypoints.json" -> "1"
        "kp_half_3_origin_keypoints.json" -> "3"
    
    Args:
        filename: 파일명
    
    Returns:
        키포인트 번호 (문자열) 또는 None
    """
    # 패턴: kp_{image_type}_{num}_origin_keypoints.json
    match = re.search(r'kp_(?:full|half)_(\d+)_origin', filename)
    if match:
        return match.group(1)
    return None


def extract_kp_num_from_generated_json(filename: str) -> Optional[str]:
    """
    생성된 모델의 JSON 파일명에서 키포인트 번호를 추출합니다.
    
    예: "full_bg_1_kp_1_keypoints.json" -> "1"
        "half_nobg_2_kp_3_keypoints.json" -> "3"
        "selfie_1_kp_2_keypoints.json" -> "2"
    
    Args:
        filename: 파일명
    
    Returns:
        키포인트 번호 (문자열) 또는 None
    """
    # 패턴: ..._kp_{num}_keypoints.json
    match = re.search(r'_kp_(\d+)_keypoints', filename)
    if match:
        return match.group(1)
    return None


def find_original_json(
    extracted_keypoints_base: Path,
    image_type: str,
    kp_num: str
) -> Optional[Path]:
    """
    original 폴더에서 해당 키포인트 번호의 JSON 파일을 찾습니다.
    
    Args:
        extracted_keypoints_base: extracted_keypoints 기본 디렉토리
        image_type: 이미지 타입 ('full' 또는 'half')
        kp_num: 키포인트 번호 (문자열)
    
    Returns:
        JSON 파일 경로 또는 None
    """
    original_dir = extracted_keypoints_base / "original" / image_type
    
    if not original_dir.exists():
        return None
    
    # 패턴: kp_{image_type}_{kp_num}_origin_keypoints.json
    pattern = f"kp_{image_type}_{kp_num}_origin_keypoints.json"
    json_path = original_dir / pattern
    
    if json_path.exists():
        return json_path
    
    # 대체 패턴 시도 (혹시 다른 형식일 경우)
    for json_file in original_dir.glob(f"*kp*{kp_num}*.json"):
        return json_file
    
    return None


def calculate_pck(
    predicted_keypoints: Dict[str, Tuple[float, float, float]],
    ground_truth_keypoints: Dict[str, Tuple[float, float, float]],
    image_width: int,
    image_height: int,
    alpha: float = 0.2
) -> float:
    """
    PCK (Percentage of Correct Keypoints)를 계산합니다.
    
    Args:
        predicted_keypoints: 예측된 키포인트 딕셔너리
        ground_truth_keypoints: 정답 키포인트 딕셔너리
        image_width: 이미지 너비
        image_height: 이미지 높이
        alpha: 임계값 (이미지 대각선 길이의 비율, 기본값: 0.2)
    
    Returns:
        PCK 점수 (0.0 ~ 1.0)
    """
    image_diagonal = np.sqrt(image_width**2 + image_height**2)
    threshold = alpha * image_diagonal
    
    correct_count = 0
    total_count = 0
    
    # 공통 키포인트만 비교
    common_keys = set(predicted_keypoints.keys()) & set(ground_truth_keypoints.keys())
    
    for key in common_keys:
        pred_x, pred_y, pred_conf = predicted_keypoints[key]
        gt_x, gt_y, gt_conf = ground_truth_keypoints[key]
        
        # confidence가 0.3 이상인 키포인트만 평가
        if pred_conf >= 0.3 and gt_conf >= 0.3:
            distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            if distance <= threshold:
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0


def calculate_pckh(
    predicted_keypoints: Dict[str, Tuple[float, float, float]],
    ground_truth_keypoints: Dict[str, Tuple[float, float, float]],
    image_width: int,
    image_height: int,
    alpha: float = 0.5
) -> float:
    """
    PCKh (PCK with head size normalization)를 계산합니다.
    
    Args:
        predicted_keypoints: 예측된 키포인트 딕셔너리
        ground_truth_keypoints: 정답 키포인트 딕셔너리
        image_width: 이미지 너비
        image_height: 이미지 높이
        alpha: 임계값 (머리 크기의 비율, 기본값: 0.5)
    
    Returns:
        PCKh 점수 (0.0 ~ 1.0)
    """
    # 머리 크기 계산 (귀 또는 눈 사이 거리)
    head_size = 50.0  # 기본값
    
    if "left_ear" in ground_truth_keypoints and "right_ear" in ground_truth_keypoints:
        left_ear = ground_truth_keypoints["left_ear"]
        right_ear = ground_truth_keypoints["right_ear"]
        if left_ear[2] > 0.3 and right_ear[2] > 0.3:
            head_size = np.sqrt(
                (left_ear[0] - right_ear[0])**2 + 
                (left_ear[1] - right_ear[1])**2
            )
    
    if head_size <= 0 and "left_eye" in ground_truth_keypoints and "right_eye" in ground_truth_keypoints:
        left_eye = ground_truth_keypoints["left_eye"]
        right_eye = ground_truth_keypoints["right_eye"]
        if left_eye[2] > 0.3 and right_eye[2] > 0.3:
            head_size = np.sqrt(
                (left_eye[0] - right_eye[0])**2 + 
                (left_eye[1] - right_eye[1])**2
            )
    
    if head_size <= 0:
        head_size = (image_width + image_height) / 10  # 기본값
    
    threshold = alpha * head_size
    
    correct_count = 0
    total_count = 0
    
    # 공통 키포인트만 비교
    common_keys = set(predicted_keypoints.keys()) & set(ground_truth_keypoints.keys())
    
    for key in common_keys:
        pred_x, pred_y, pred_conf = predicted_keypoints[key]
        gt_x, gt_y, gt_conf = ground_truth_keypoints[key]
        
        # confidence가 0.3 이상인 키포인트만 평가
        if pred_conf >= 0.3 and gt_conf >= 0.3:
            distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            if distance <= threshold:
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0


def evaluate_extracted_keypoints(
    model_name: str,
    prompt_version: str,
    extracted_keypoints_base: Path = Path("extracted_keypoints"),
    output_dir: Path = Path("evaluations/pck")
) -> Dict:
    """
    extracted_keypoints의 original과 생성 모델 간 PCK/PCKh 평가를 수행합니다.
    
    Args:
        model_name: 모델 이름 (예: 'nano_banana')
        prompt_version: 프롬프트 버전 (예: 'short', 'medium', 'long')
        extracted_keypoints_base: extracted_keypoints 기본 디렉토리
        output_dir: 결과 출력 디렉토리
    
    Returns:
        평가 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"PCK/PCKh 평가 시작: {model_name}/{prompt_version}")
    print(f"{'='*60}\n")
    
    # original 폴더 확인
    original_base = extracted_keypoints_base / "original"
    if not original_base.exists():
        print(f"⚠️  original 폴더를 찾을 수 없습니다: {original_base}")
        return {}
    
    # 생성 모델 폴더 확인
    generated_dir = extracted_keypoints_base / model_name / prompt_version
    if not generated_dir.exists():
        print(f"⚠️  생성 모델 폴더를 찾을 수 없습니다: {generated_dir}")
        return {}
    
    # 이미지 타입별로 처리 (original에는 full, half만 있음)
    image_types = ["full", "half"]
    results = []
    
    for image_type in image_types:
        original_type_dir = original_base / image_type
        generated_type_dir = generated_dir / image_type
        
        if not original_type_dir.exists():
            print(f"⚠️  original/{image_type} 폴더를 찾을 수 없습니다: {original_type_dir}")
            continue
        
        if not generated_type_dir.exists():
            print(f"⚠️  {model_name}/{prompt_version}/{image_type} 폴더를 찾을 수 없습니다: {generated_type_dir}")
            continue
        
        print(f"📁 이미지 타입: {image_type}")
        
        # full, half는 bg, no_bg 구분 (selfie는 별도 처리)
        bg_types = ["bg", "no_bg"]
        
        for bg_type in bg_types:
            bg_dir = generated_type_dir / bg_type
            
            if not bg_dir.exists():
                print(f"  ⚠️  {bg_type} 폴더를 찾을 수 없습니다: {bg_dir}")
                continue
            
            print(f"  📂 배경 타입: {bg_type}")
            
            # JSON 파일 찾기
            json_files = list(bg_dir.glob("*_keypoints.json"))
            
            print(f"    📸 발견된 JSON 파일: {len(json_files)}개")
            
            for json_path in json_files:
                try:
                    # 생성된 모델의 JSON에서 키포인트 번호 추출
                    kp_num = extract_kp_num_from_generated_json(json_path.name)
                    
                    if kp_num is None:
                        print(f"    ⚠️  키포인트 번호를 추출할 수 없습니다: {json_path.name}")
                        continue
                    
                    # original 폴더에서 대응하는 JSON 찾기
                    original_json_path = find_original_json(
                        extracted_keypoints_base, image_type, kp_num
                    )
                    
                    if original_json_path is None:
                        print(f"    ⚠️  original JSON을 찾을 수 없습니다: kp_{image_type}_{kp_num}")
                        continue
                    
                    # 키포인트 로드
                    predicted_keypoints = load_keypoints_from_json(json_path)
                    ground_truth_keypoints = load_keypoints_from_json(original_json_path)
                    
                    if predicted_keypoints is None:
                        print(f"    ⚠️  생성 모델 키포인트를 로드할 수 없습니다: {json_path.name}")
                        continue
                    
                    if ground_truth_keypoints is None:
                        print(f"    ⚠️  original 키포인트를 로드할 수 없습니다: {original_json_path.name}")
                        continue
                    
                    # 이미지 크기 가져오기 (생성된 모델의 JSON과 같은 이름의 이미지 파일 찾기)
                    # JSON 파일명에서 이미지 파일명 생성
                    image_stem = json_path.stem.replace("_keypoints", "")
                    image_path = json_path.parent / f"{image_stem}.png"
                    if not image_path.exists():
                        image_path = json_path.parent / f"{image_stem}.jpg"
                    if not image_path.exists():
                        image_path = json_path.parent / f"{image_stem}.jpeg"
                    
                    if image_path.exists():
                        pred_image = Image.open(image_path)
                        pred_width, pred_height = pred_image.size
                    else:
                        # 이미지가 없으면 original 이미지 크기 사용
                        original_image_stem = original_json_path.stem.replace("_keypoints", "")
                        original_image_path = original_json_path.parent / f"{original_image_stem}.png"
                        if not original_image_path.exists():
                            original_image_path = original_json_path.parent / f"{original_image_stem}.jpg"
                        if original_image_path.exists():
                            pred_image = Image.open(original_image_path)
                            pred_width, pred_height = pred_image.size
                        else:
                            # 기본값 사용
                            pred_width, pred_height = 1920, 1080
                    
                    # PCK 계산 (여러 alpha 값)
                    pck_01 = calculate_pck(
                        predicted_keypoints, ground_truth_keypoints,
                        pred_width, pred_height, alpha=0.1
                    )
                    pck_02 = calculate_pck(
                        predicted_keypoints, ground_truth_keypoints,
                        pred_width, pred_height, alpha=0.2
                    )
                    
                    # PCKh 계산
                    pckh_05 = calculate_pckh(
                        predicted_keypoints, ground_truth_keypoints,
                        pred_width, pred_height, alpha=0.5
                    )
                    
                    result = {
                        "generated_json": str(json_path),
                        "original_json": str(original_json_path),
                        "image_type": image_type,
                        "bg_type": bg_type,
                        "kp_num": kp_num,
                        "pck_0.1": round(pck_01, 4),
                        "pck_0.2": round(pck_02, 4),
                        "pckh_0.5": round(pckh_05, 4),
                    }
                    
                    results.append(result)
                    print(f"    ✓ {json_path.name} ↔ {original_json_path.name}: PCK@0.2={pck_02:.4f}, PCKh@0.5={pckh_05:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ 오류 발생 ({json_path.name}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # selfie 처리 (별도)
        if image_type == "half":  # selfie는 half와 함께 처리
            selfie_dir = generated_dir / "selfie"
            if selfie_dir.exists():
                print(f"\n📁 이미지 타입: selfie")
                print(f"  📂 배경 타입: 없음 (selfie)")
                
                json_files = list(selfie_dir.glob("*_keypoints.json"))
                print(f"    📸 발견된 JSON 파일: {len(json_files)}개")
                
                for json_path in json_files:
                    try:
                        kp_num = extract_kp_num_from_generated_json(json_path.name)
                        
                        if kp_num is None:
                            print(f"    ⚠️  키포인트 번호를 추출할 수 없습니다: {json_path.name}")
                            continue
                        
                        # selfie는 half와 매칭
                        original_json_path = find_original_json(
                            extracted_keypoints_base, "half", kp_num
                        )
                        
                        if original_json_path is None:
                            print(f"    ⚠️  original JSON을 찾을 수 없습니다: kp_half_{kp_num}")
                            continue
                        
                        predicted_keypoints = load_keypoints_from_json(json_path)
                        ground_truth_keypoints = load_keypoints_from_json(original_json_path)
                        
                        if predicted_keypoints is None or ground_truth_keypoints is None:
                            continue
                        
                        # 이미지 크기 가져오기
                        image_stem = json_path.stem.replace("_keypoints", "")
                        image_path = json_path.parent / f"{image_stem}.png"
                        if not image_path.exists():
                            image_path = json_path.parent / f"{image_stem}.jpg"
                        
                        if image_path.exists():
                            pred_image = Image.open(image_path)
                            pred_width, pred_height = pred_image.size
                        else:
                            pred_width, pred_height = 1920, 1080
                        
                        pck_01 = calculate_pck(
                            predicted_keypoints, ground_truth_keypoints,
                            pred_width, pred_height, alpha=0.1
                        )
                        pck_02 = calculate_pck(
                            predicted_keypoints, ground_truth_keypoints,
                            pred_width, pred_height, alpha=0.2
                        )
                        pckh_05 = calculate_pckh(
                            predicted_keypoints, ground_truth_keypoints,
                            pred_width, pred_height, alpha=0.5
                        )
                        
                        result = {
                            "generated_json": str(json_path),
                            "original_json": str(original_json_path),
                            "image_type": "selfie",
                            "bg_type": None,
                            "kp_num": kp_num,
                            "pck_0.1": round(pck_01, 4),
                            "pck_0.2": round(pck_02, 4),
                            "pckh_0.5": round(pckh_05, 4),
                        }
                        
                        results.append(result)
                        print(f"    ✓ {json_path.name} ↔ {original_json_path.name}: PCK@0.2={pck_02:.4f}, PCKh@0.5={pckh_05:.4f}")
                        
                    except Exception as e:
                        print(f"    ❌ 오류 발생 ({json_path.name}): {e}")
                        continue
    
    # 통계 계산
    if results:
        pck_01_scores = [r["pck_0.1"] for r in results]
        pck_02_scores = [r["pck_0.2"] for r in results]
        pckh_05_scores = [r["pckh_0.5"] for r in results]
        
        statistics = {
            "mean_pck_0.1": round(float(np.mean(pck_01_scores)), 4),
            "mean_pck_0.2": round(float(np.mean(pck_02_scores)), 4),
            "mean_pckh_0.5": round(float(np.mean(pckh_05_scores)), 4),
            "std_pck_0.1": round(float(np.std(pck_01_scores)), 4),
            "std_pck_0.2": round(float(np.std(pck_02_scores)), 4),
            "std_pckh_0.5": round(float(np.std(pckh_05_scores)), 4),
            "total_images": len(results),
        }
    else:
        statistics = {
            "mean_pck_0.1": None,
            "mean_pck_0.2": None,
            "mean_pckh_0.5": None,
            "std_pck_0.1": None,
            "std_pck_0.2": None,
            "std_pckh_0.5": None,
            "total_images": 0,
        }
    
    # 결과 저장
    output_data = {
        "model": model_name,
        "prompt_version": prompt_version,
        "statistics": statistics,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = output_dir / model_name / f"{prompt_version}_pck_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    if statistics["total_images"] > 0:
        print(f"\n📊 통계:")
        print(f"  평균 PCK@0.1: {statistics['mean_pck_0.1']:.4f}")
        print(f"  평균 PCK@0.2: {statistics['mean_pck_0.2']:.4f}")
        print(f"  평균 PCKh@0.5: {statistics['mean_pckh_0.5']:.4f}")
        print(f"  총 이미지 수: {statistics['total_images']}")
    
    print(f"\n{'='*60}")
    print("평가 완료")
    print(f"{'='*60}\n")
    
    return output_data


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="extracted_keypoints의 original과 생성 모델 간 PCK/PCKh 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # nano_banana 모델의 short 프롬프트 평가
  python metrics/pck/evaluate_extracted_keypoints.py --model nano_banana --prompt short
  
  # 여러 프롬프트 버전 평가
  python metrics/pck/evaluate_extracted_keypoints.py --model nano_banana --prompt short medium long
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="모델 이름 (예: nano_banana)"
    )
    
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=["short", "medium", "long"],
        help="프롬프트 버전 (기본값: short medium long)"
    )
    
    parser.add_argument(
        "--extracted_keypoints_dir",
        type=str,
        default="extracted_keypoints",
        help="extracted_keypoints 기본 디렉토리 (기본값: extracted_keypoints)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluations/pck",
        help="결과 출력 디렉토리 (기본값: evaluations/pck)"
    )
    
    args = parser.parse_args()
    
    # 평가 실행
    for prompt_version in args.prompt:
        evaluate_extracted_keypoints(
            model_name=args.model,
            prompt_version=prompt_version,
            extracted_keypoints_base=Path(args.extracted_keypoints_dir),
            output_dir=Path(args.output_dir)
        )


if __name__ == "__main__":
    main()
