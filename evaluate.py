"""
평가 메트릭 실행 스크립트

생성된 이미지들을 평가하여 LPIPS, PCK 등의 메트릭을 계산합니다.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from utils import get_image_files, get_keypoint_path
from metrics.lpips import calculate_lpips, evaluate_lpips_batch
from metrics.pck import PCKEvaluator


# 모델 및 프롬프트 설정
PROMPT_VERSIONS = ["short", "medium", "long"]
IMAGE_TYPES = ["full", "half", "selfie"]
KP_NUMS = [1, 2, 3]


def find_result_image(
    model_name: str,
    prompt_version: str,
    image_type: str,
    bg_type: Optional[str],
    image_name: str,
    kp_num: int
) -> Optional[Path]:
    """
    생성된 결과 이미지 경로를 찾습니다.
    
    Args:
        model_name: 모델 이름
        prompt_version: 프롬프트 버전
        image_type: 이미지 타입
        bg_type: 배경 타입 (None이면 selfie)
        image_name: 원본 이미지 이름 (확장자 제외)
        kp_num: 키포인트 번호
    
    Returns:
        결과 이미지 경로 또는 None
    """
    # 확장자 목록
    extensions = [".jpg", ".jpeg", ".png"]
    
    if bg_type:
        base_dir = Path(f"results/{model_name}/{prompt_version}/{image_type}/{bg_type}")
        base_name = f"{image_name}_kp_{kp_num}"
    else:
        base_dir = Path(f"results/{model_name}/{prompt_version}/{image_type}")
        base_name = f"{image_name}_kp_{kp_num}"
    
    # 여러 확장자 시도
    for ext in extensions:
        result_path = base_dir / f"{base_name}{ext}"
        if result_path.exists():
            return result_path
    
    return None


def evaluate_lpips_metric(
    model_name: str,
    prompt_versions: List[str],
    image_types: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    LPIPS 메트릭을 계산합니다.
    
    Args:
        model_name: 모델 이름
        prompt_versions: 평가할 프롬프트 버전 리스트
        image_types: 평가할 이미지 타입 리스트 (None이면 모든 타입)
        output_dir: 결과 저장 디렉토리
    
    Returns:
        평가 결과 딕셔너리
    """
    if image_types is None:
        image_types = IMAGE_TYPES
    
    if output_dir is None:
        output_dir = Path("evaluations/lpips")
    
    all_results = {}
    
    for prompt_version in prompt_versions:
        print(f"\n{'='*60}")
        print(f"LPIPS 평가: {model_name} / {prompt_version}")
        print(f"{'='*60}")
        
        original_images = []
        generated_images = []
        image_info = []  # 각 이미지 쌍의 메타데이터
        
        # 각 이미지 타입별로 처리
        for image_type in image_types:
            print(f"\n  이미지 타입: {image_type}")
            
            if image_type == "selfie":
                # selfie는 bg_type 구분 없음
                image_files = get_image_files(image_type)
                bg_type = None
                
                for image_path in image_files:
                    image_name = image_path.stem  # 확장자 제외
                    
                    for kp_num in KP_NUMS:
                        result_path = find_result_image(
                            model_name, prompt_version, image_type,
                            bg_type, image_name, kp_num
                        )
                        
                        if result_path and result_path.exists():
                            original_images.append(image_path)
                            generated_images.append(result_path)
                            image_info.append({
                                "image_type": image_type,
                                "bg_type": "none",
                                "image_name": image_name,
                                "kp_num": kp_num,
                                "original_path": str(image_path),
                                "generated_path": str(result_path)
                            })
            else:
                # full, half는 bg/no_bg 구분
                for bg_type in ["bg", "no_bg"]:
                    image_files = get_image_files(image_type, bg_type)
                    
                    for image_path in image_files:
                        image_name = image_path.stem  # 확장자 제외
                        
                        for kp_num in KP_NUMS:
                            result_path = find_result_image(
                                model_name, prompt_version, image_type,
                                bg_type, image_name, kp_num
                            )
                            
                            if result_path and result_path.exists():
                                original_images.append(image_path)
                                generated_images.append(result_path)
                                image_info.append({
                                    "image_type": image_type,
                                    "bg_type": bg_type,
                                    "image_name": image_name,
                                    "kp_num": kp_num,
                                    "original_path": str(image_path),
                                    "generated_path": str(result_path)
                                })
        
        if not original_images:
            print(f"  ⚠️  평가할 이미지가 없습니다.")
            continue
        
        print(f"  📊 평가할 이미지 쌍: {len(original_images)}개")
        
        # LPIPS 일괄 평가
        results = evaluate_lpips_batch(
            original_images=original_images,
            generated_images=generated_images,
            output_path=None,  # 여기서는 저장하지 않음
            net='vgg',
            device=None
        )
        
        # 결과에 메타데이터 추가
        for i, result in enumerate(results):
            result.update(image_info[i])
            result["model"] = model_name
            result["prompt_version"] = prompt_version
        
        # 통계 계산
        lpips_scores = [r["lpips_score"] for r in results if r.get("lpips_score") is not None]
        
        if lpips_scores:
            statistics = {
                "mean": float(np.mean(lpips_scores)),
                "std": float(np.std(lpips_scores)),
                "min": float(np.min(lpips_scores)),
                "max": float(np.max(lpips_scores)),
                "total_images": len(results),
                "valid_images": len(lpips_scores)
            }
        else:
            statistics = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "total_images": len(results),
                "valid_images": 0
            }
        
        # 결과 저장
        output_data = {
            "model": model_name,
            "prompt_version": prompt_version,
            "statistics": statistics,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = output_dir / model_name / f"{prompt_version}_lpips_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 결과 저장: {output_file}")
        print(f"  📈 평균 LPIPS: {statistics['mean']:.4f}" if statistics['mean'] else "  📈 평균 LPIPS: N/A")
        
        all_results[prompt_version] = output_data
    
    return all_results


def evaluate_pck_metric(
    model_name: str,
    prompt_versions: List[str],
    image_types: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    openpose_path: Optional[str] = None,
    model_folder: Optional[str] = None
) -> Dict:
    """
    PCK/PCKh 메트릭을 계산합니다.
    
    Args:
        model_name: 모델 이름
        prompt_versions: 평가할 프롬프트 버전 리스트
        image_types: 평가할 이미지 타입 리스트 (None이면 모든 타입)
        output_dir: 결과 저장 디렉토리
        openpose_path: OpenPose 빌드 경로
        model_folder: OpenPose 모델 폴더 경로
    
    Returns:
        평가 결과 딕셔너리
    """
    if image_types is None:
        image_types = IMAGE_TYPES
    
    if output_dir is None:
        output_dir = Path("evaluations/pck")
    
    evaluator = PCKEvaluator(openpose_path=openpose_path, model_folder=model_folder)
    all_results = {}
    
    for prompt_version in prompt_versions:
        print(f"\n{'='*60}")
        print(f"PCK/PCKh 평가: {model_name} / {prompt_version}")
        print(f"{'='*60}")
        
        results = []
        
        # 각 이미지 타입별로 처리
        for image_type in image_types:
            print(f"\n  이미지 타입: {image_type}")
            
            if image_type == "selfie":
                # selfie는 bg_type 구분 없음, half 키포인트 사용
                image_files = get_image_files(image_type)
                bg_type = None
                kp_type = "half"
                
                for image_path in image_files:
                    image_name = image_path.stem
                    
                    for kp_num in KP_NUMS:
                        result_path = find_result_image(
                            model_name, prompt_version, image_type,
                            bg_type, image_name, kp_num
                        )
                        
                        if result_path and result_path.exists():
                            # 참조 키포인트 이미지 경로
                            ref_keypoint_path = get_keypoint_path(kp_type, kp_num)
                            
                            if ref_keypoint_path.exists():
                                print(f"    평가 중: {image_name}_kp_{kp_num}")
                                
                                try:
                                    pck_result = evaluator.evaluate(
                                        generated_image_path=result_path,
                                        reference_keypoint_path=ref_keypoint_path,
                                        image_type=kp_type
                                    )
                                    
                                    pck_result.update({
                                        "image_type": image_type,
                                        "bg_type": "none",
                                        "image_name": image_name,
                                        "kp_num": kp_num,
                                        "generated_path": str(result_path),
                                        "reference_keypoint_path": str(ref_keypoint_path),
                                        "model": model_name,
                                        "prompt_version": prompt_version
                                    })
                                    
                                    results.append(pck_result)
                                except Exception as e:
                                    print(f"    ❌ 오류: {str(e)}")
                                    results.append({
                                        "image_type": image_type,
                                        "bg_type": "none",
                                        "image_name": image_name,
                                        "kp_num": kp_num,
                                        "error": str(e),
                                        "model": model_name,
                                        "prompt_version": prompt_version
                                    })
            else:
                # full, half는 bg/no_bg 구분
                for bg_type in ["bg", "no_bg"]:
                    image_files = get_image_files(image_type, bg_type)
                    kp_type = image_type
                    
                    for image_path in image_files:
                        image_name = image_path.stem
                        
                        for kp_num in KP_NUMS:
                            result_path = find_result_image(
                                model_name, prompt_version, image_type,
                                bg_type, image_name, kp_num
                            )
                            
                            if result_path and result_path.exists():
                                # 참조 키포인트 이미지 경로
                                ref_keypoint_path = get_keypoint_path(kp_type, kp_num)
                                
                                if ref_keypoint_path.exists():
                                    print(f"    평가 중: {image_name}_kp_{kp_num}")
                                    
                                    try:
                                        pck_result = evaluator.evaluate(
                                            generated_image_path=result_path,
                                            reference_keypoint_path=ref_keypoint_path,
                                            image_type=kp_type
                                        )
                                        
                                        pck_result.update({
                                            "image_type": image_type,
                                            "bg_type": bg_type,
                                            "image_name": image_name,
                                            "kp_num": kp_num,
                                            "generated_path": str(result_path),
                                            "reference_keypoint_path": str(ref_keypoint_path),
                                            "model": model_name,
                                            "prompt_version": prompt_version
                                        })
                                        
                                        results.append(pck_result)
                                    except Exception as e:
                                        print(f"    ❌ 오류: {str(e)}")
                                        results.append({
                                            "image_type": image_type,
                                            "bg_type": bg_type,
                                            "image_name": image_name,
                                            "kp_num": kp_num,
                                            "error": str(e),
                                            "model": model_name,
                                            "prompt_version": prompt_version
                                        })
        
        if not results:
            print(f"  ⚠️  평가할 이미지가 없습니다.")
            continue
        
        # 통계 계산
        valid_results = [r for r in results if "error" not in r and r.get("pck_0.1") is not None]
        
        if valid_results:
            pck_01_scores = [r["pck_0.1"] for r in valid_results]
            pck_02_scores = [r["pck_0.2"] for r in valid_results]
            pckh_05_scores = [r["pckh_0.5"] for r in valid_results]
            mean_errors = [r["mean_keypoint_error"] for r in valid_results]
            
            statistics = {
                "total_images": len(results),
                "valid_images": len(valid_results),
                "pck_0.1": {
                    "mean": float(np.mean(pck_01_scores)),
                    "std": float(np.std(pck_01_scores)),
                    "min": float(np.min(pck_01_scores)),
                    "max": float(np.max(pck_01_scores))
                },
                "pck_0.2": {
                    "mean": float(np.mean(pck_02_scores)),
                    "std": float(np.std(pck_02_scores)),
                    "min": float(np.min(pck_02_scores)),
                    "max": float(np.max(pck_02_scores))
                },
                "pckh_0.5": {
                    "mean": float(np.mean(pckh_05_scores)),
                    "std": float(np.std(pckh_05_scores)),
                    "min": float(np.min(pckh_05_scores)),
                    "max": float(np.max(pckh_05_scores))
                },
                "mean_keypoint_error": {
                    "mean": float(np.mean(mean_errors)),
                    "std": float(np.std(mean_errors)),
                    "min": float(np.min(mean_errors)),
                    "max": float(np.max(mean_errors))
                }
            }
        else:
            statistics = {
                "total_images": len(results),
                "valid_images": 0
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
        
        print(f"  💾 결과 저장: {output_file}")
        if valid_results:
            print(f"  📈 평균 PCK@0.1: {statistics['pck_0.1']['mean']:.4f}")
            print(f"  📈 평균 PCK@0.2: {statistics['pck_0.2']['mean']:.4f}")
            print(f"  📈 평균 PCKh@0.5: {statistics['pckh_0.5']['mean']:.4f}")
        
        all_results[prompt_version] = output_data
    
    return all_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="평가 메트릭 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # LPIPS 평가
  python evaluate.py --metric lpips --model nano_banana --prompt short
  
  # PCK 평가
  python evaluate.py --metric pck --model nano_banana --prompt short
  
  # 여러 프롬프트 버전 평가
  python evaluate.py --metric lpips --model nano_banana --prompt short medium long
  
  # 특정 이미지 타입만 평가
  python evaluate.py --metric lpips --model nano_banana --prompt short --image_type full
        """
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["lpips", "pck", "all"],
        help="평가할 메트릭 (lpips, pck, all)"
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
        choices=PROMPT_VERSIONS,
        default=PROMPT_VERSIONS,
        help=f"평가할 프롬프트 버전 (기본값: 모든 버전). 선택 가능: {', '.join(PROMPT_VERSIONS)}"
    )
    
    parser.add_argument(
        "--image_type",
        nargs="+",
        choices=IMAGE_TYPES,
        default=None,
        help=f"평가할 이미지 타입 (기본값: 모든 타입). 선택 가능: {', '.join(IMAGE_TYPES)}"
    )
    
    parser.add_argument(
        "--openpose_path",
        type=str,
        default=None,
        help="OpenPose 빌드 경로 (PCK 평가 시 필요)"
    )
    
    parser.add_argument(
        "--openpose_model_folder",
        type=str,
        default=None,
        help="OpenPose 모델 폴더 경로 (PCK 평가 시 필요)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("평가 메트릭 실행")
    print("="*60)
    print(f"메트릭: {args.metric}")
    print(f"모델: {args.model}")
    print(f"프롬프트 버전: {', '.join(args.prompt)}")
    if args.image_type:
        print(f"이미지 타입: {', '.join(args.image_type)}")
    print("="*60)
    
    # 메트릭 실행
    if args.metric == "lpips" or args.metric == "all":
        evaluate_lpips_metric(
            model_name=args.model,
            prompt_versions=args.prompt,
            image_types=args.image_type
        )
    
    if args.metric == "pck" or args.metric == "all":
        evaluate_pck_metric(
            model_name=args.model,
            prompt_versions=args.prompt,
            image_types=args.image_type,
            openpose_path=args.openpose_path,
            model_folder=args.openpose_model_folder
        )
    
    print("\n" + "="*60)
    print("평가 완료")
    print("="*60)


if __name__ == "__main__":
    main()

