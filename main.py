"""
테스트 케이스 자동화 스크립트

3개 모델 × 3개 프롬프트 버전 = 9개 조합에 대해
각 이미지 타입(full, half, selfie)별로 키포인트를 적용하여 이미지를 생성합니다.
"""

import os
import json
import argparse
from pathlib import Path

# .env 파일에서 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv가 설치되지 않은 경우 경고만 출력
    pass

from models import NanoBanana, StableDiffusion, QwenControlnet
from utils import (
    load_image,
    get_image_files,
    get_keypoint_path,
    create_output_path
)
from metrics.timer import elapse_time

# 모델 및 프롬프트 설정
MODELS = {
    "nano_banana": NanoBanana,
    "stable_diffusion": StableDiffusion,
    "qwen_controlnet": QwenControlnet
}

PROMPT_VERSIONS = ["short", "medium", "long"]
IMAGE_TYPES = ["full", "half", "selfie"]
KP_NUMS = [1, 2, 3]


def get_prompt_path(model_name, prompt_version):
    """프롬프트 파일 경로를 반환합니다."""
    return Path(f"prompts/{model_name}/{prompt_version}.txt")


def process_single_case(model_name, prompt_version, image_type, bg_type, image_path, kp_num):
    """
    단일 테스트 케이스를 처리합니다.
    
    Args:
        model_name: 모델 이름
        prompt_version: 프롬프트 버전 ('short', 'medium', 'long')
        image_type: 이미지 타입 ('full', 'half', 'selfie')
        bg_type: 배경 타입 ('bg', 'no_bg', None)
        image_path: 원본 이미지 경로
        kp_num: 키포인트 번호 (1, 2, 3)
    
    Returns:
        생성된 이미지 경로 또는 None (실패 시)
    """
    try:
        # 모델 초기화
        prompt_path = get_prompt_path(model_name, prompt_version)
        if not prompt_path.exists():
            print(f"⚠️  프롬프트 파일이 없습니다: {prompt_path}")
            return None
        
        model_class = MODELS[model_name]
        model = model_class(str(prompt_path))
        
        # 이미지 로드
        input_image = load_image(image_path)
        
        # 키포인트 로드 (selfie는 half 키포인트 사용)
        kp_type = "half" if image_type == "selfie" else image_type
        kp_path = get_keypoint_path(kp_type, kp_num)
        if not kp_path.exists():
            print(f"⚠️  키포인트 파일이 없습니다: {kp_path}")
            return None
        
        pose_image = load_image(kp_path)
        
        # 출력 경로 생성
        image_name = image_path.stem  # 확장자 제외한 파일명
        output_path = create_output_path(
            model_name, prompt_version, image_type, bg_type, image_name, kp_num
        )
        
        # 이미지 생성
        result = model.generate(input_image, pose_image, output_path)
        
        # 결과가 dict인 경우 (시간/토큰 정보 포함)와 str인 경우 (기존 형식) 모두 처리
        if isinstance(result, dict):
            return result
        else:
            # 기존 형식 (문자열만 반환) - 호환성을 위해 dict로 변환
            return {
                "saved_path": result,
                "generation_time": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
        
    except Exception as e:
        print(f"❌ 오류 발생: {model_name}/{prompt_version}/{image_type}/{bg_type}/{image_path.name}_kp_{kp_num}")
        print(f"   오류 내용: {str(e)}")
        return None


def save_token_time_json(model_name, prompt_version, results_dict):
    """
    시간과 토큰 정보를 JSON 파일로 저장합니다.
    
    Args:
        model_name: 모델 이름
        prompt_version: 프롬프트 버전
        results_dict: {파일경로: {generation_time, total_tokens, input_tokens, output_tokens}} 형태의 딕셔너리
    """
    output_dir = Path(f"results/{model_name}/{prompt_version}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "token_time_result.json"
    
    # 소수점 1자리로 변환
    formatted_results = {}
    for file_path, info in results_dict.items():
        formatted_results[file_path] = {
            "generation_time": round(info["generation_time"], 1),
            "total_tokens": info["total_tokens"],
            "input_tokens": info["input_tokens"],
            "output_tokens": info["output_tokens"]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 토큰/시간 정보 저장: {output_file}")


def run_all_tests(model_names=None, prompt_versions=None, limit_images=None):
    """
    테스트 케이스를 실행합니다.
    
    Args:
        model_names: 실행할 모델 이름 리스트 (None이면 모든 모델)
        prompt_versions: 실행할 프롬프트 버전 리스트 (None이면 모든 버전)
        limit_images: 테스트할 이미지 수 제한 (None이면 모든 이미지)
    """
    
    # 기본값 설정
    if model_names is None:
        model_names = list(MODELS.keys())
    if prompt_versions is None:
        prompt_versions = PROMPT_VERSIONS
    
    # 유효성 검사
    for model_name in model_names:
        if model_name not in MODELS:
            raise ValueError(f"잘못된 모델 이름: {model_name}. 사용 가능한 모델: {list(MODELS.keys())}")
    
    for prompt_version in prompt_versions:
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"잘못된 프롬프트 버전: {prompt_version}. 사용 가능한 버전: {PROMPT_VERSIONS}")
    
    total_cases = 0
    successful_cases = 0
    failed_cases = 0
    
    # 각 모델별로 처리
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"모델: {model_name}")
        print(f"{'='*60}")
        
        # 각 프롬프트 버전별로 처리
        for prompt_version in prompt_versions:
            print(f"\n  프롬프트 버전: {prompt_version}")
            
            # 토큰/시간 정보 수집용 딕셔너리
            token_time_results = {}
            
            # 각 모델/프롬프트 조합별로 시간 측정
            with elapse_time():
                image_count = 0  # 이미지 카운터 (limit_images 제한용)
                
                # 각 이미지 타입별로 처리
                for image_type in IMAGE_TYPES:
                    if limit_images and image_count >= limit_images:
                        print(f"    ⚠️  이미지 제한 도달 ({limit_images}개), 중단")
                        break
                    
                    print(f"    이미지 타입: {image_type}")
                    
                    # 이미지 파일 가져오기
                    if image_type == "selfie":
                        # selfie는 bg_type 구분 없음
                        image_files = get_image_files(image_type)
                        bg_type = None
                        
                        # 각 이미지에 대해 처리
                        for image_path in image_files:
                            if limit_images and image_count >= limit_images:
                                break
                            
                            # 각 키포인트에 대해 처리
                            for kp_num in KP_NUMS:
                                if limit_images and image_count >= limit_images:
                                    break
                                
                                total_cases += 1
                                image_count += 1
                                
                                case_info = (
                                    f"{model_name}/{prompt_version}/{image_type}/"
                                    f"none/{image_path.name}_kp_{kp_num}"
                                )
                                
                                result = process_single_case(
                                    model_name, prompt_version, image_type, 
                                    bg_type, image_path, kp_num
                                )
                                
                                if result and isinstance(result, dict) and result.get("saved_path"):
                                    successful_cases += 1
                                    print(f"      ✓ {case_info}")
                                    
                                    # 토큰/시간 정보 저장
                                    file_path = result["saved_path"]
                                    token_time_results[file_path] = {
                                        "generation_time": result.get("generation_time", 0),
                                        "total_tokens": result.get("total_tokens", 0),
                                        "input_tokens": result.get("input_tokens", 0),
                                        "output_tokens": result.get("output_tokens", 0)
                                    }
                                else:
                                    failed_cases += 1
                                    print(f"      ✗ {case_info}")
                    else:
                        # full, half는 bg/no_bg 구분
                        for bg_type in ["bg", "no_bg"]:
                            if limit_images and image_count >= limit_images:
                                break
                            
                            image_files = get_image_files(image_type, bg_type)
                            
                            # 각 이미지에 대해 처리
                            for image_path in image_files:
                                if limit_images and image_count >= limit_images:
                                    break
                                
                                # 각 키포인트에 대해 처리
                                for kp_num in KP_NUMS:
                                    if limit_images and image_count >= limit_images:
                                        break
                                    
                                    total_cases += 1
                                    image_count += 1
                                    
                                    case_info = (
                                        f"{model_name}/{prompt_version}/{image_type}/"
                                        f"{bg_type}/{image_path.name}_kp_{kp_num}"
                                    )
                                    
                                    result = process_single_case(
                                        model_name, prompt_version, image_type, 
                                        bg_type, image_path, kp_num
                                    )
                                    
                                    if result and isinstance(result, dict) and result.get("saved_path"):
                                        successful_cases += 1
                                        print(f"      ✓ {case_info}")
                                        
                                        # 토큰/시간 정보 저장
                                        file_path = result["saved_path"]
                                        token_time_results[file_path] = {
                                            "generation_time": result.get("generation_time", 0),
                                            "total_tokens": result.get("total_tokens", 0),
                                            "input_tokens": result.get("input_tokens", 0),
                                            "output_tokens": result.get("output_tokens", 0)
                                        }
                                    else:
                                        failed_cases += 1
                                        print(f"      ✗ {case_info}")
            
            # 토큰/시간 정보 JSON 저장
            if token_time_results:
                save_token_time_json(model_name, prompt_version, token_time_results)
                    
    # 최종 통계 출력
    print(f"\n{'='*60}")
    print("테스트 완료")
    print(f"{'='*60}")
    print(f"총 테스트 케이스: {total_cases}")
    print(f"성공: {successful_cases}")
    print(f"실패: {failed_cases}")
    print(f"성공률: {successful_cases/total_cases*100:.2f}%" if total_cases > 0 else "N/A")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="테스트 케이스 자동화 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # nano_banana 모델의 short 프롬프트만 실행
  python main.py --model nano_banana --prompt short
  
  # 여러 모델과 프롬프트 조합 실행
  python main.py --model nano_banana stable_diffusion --prompt short medium
  
  # 모든 모델과 프롬프트 실행 (인자 없이)
  python main.py
        """
    )
    
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help=f"실행할 모델 이름 (기본값: 모든 모델). 선택 가능: {', '.join(MODELS.keys())}"
    )
    
    parser.add_argument(
        "--prompt",
        nargs="+",
        choices=PROMPT_VERSIONS,
        default=None,
        help=f"실행할 프롬프트 버전 (기본값: 모든 버전). 선택 가능: {', '.join(PROMPT_VERSIONS)}"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트할 이미지 수를 제한합니다 (예: 3)."
    )
    
    args = parser.parse_args()
    
    # results 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    
    # 실행 정보 출력
    model_names = args.model if args.model else list(MODELS.keys())
    prompt_versions = args.prompt if args.prompt else PROMPT_VERSIONS
    
    print("테스트 케이스 자동화 시작")
    print(f"모델: {', '.join(model_names)}")
    print(f"프롬프트 버전: {', '.join(prompt_versions)}")
    print(f"이미지 타입: {len(IMAGE_TYPES)}개 ({', '.join(IMAGE_TYPES)})")
    print(f"키포인트 버전: {len(KP_NUMS)}개")
    if args.limit:
        print(f"⚠️  이미지 제한: {args.limit}개만 테스트")
    print()
    
    # elapse_time으로 전체 실행 시간 측정
    with elapse_time():
        run_all_tests(
            model_names=model_names, 
            prompt_versions=prompt_versions,
            limit_images=args.limit
        )


if __name__ == "__main__":
    main()
