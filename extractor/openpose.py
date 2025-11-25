"""
키포인트 추출 모듈 (OpenPose 기반)

OpenPose를 사용하여 이미지에서 키포인트를 추출합니다.
두 가지 방식을 지원합니다:
1. pyopenpose Python API (기존 방식)
2. OpenPose 실행 파일 (subprocess 방식)
"""

import json
import numpy as np
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2

# OpenPose BODY_25 모델의 연결 관계
BODY_PAIRS = [
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
    [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]
]

FACE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20], [20, 21],
    [22, 23], [23, 24], [24, 25], [25, 26],
    [27, 28], [28, 29], [29, 30],
    [31, 32], [32, 33], [33, 34], [34, 35],
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
    [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
    [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55],
    [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
]

HAND_PAIRS = [
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
    [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20]
]

# pyopenpose Python API 지원 여부 확인
try:
    import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False


class KeypointExtractor:
    """
    OpenPose를 사용하여 이미지에서 키포인트를 추출하는 클래스 (pyopenpose Python API 사용)
    """
    
    # OpenPose BODY_25 키포인트 인덱스 (25개 키포인트)
    KEYPOINT_NAMES = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
        "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel"
    ]
    
    # 상반신 키포인트 인덱스 (half/selfie용) - 상반신 + 얼굴
    UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
    
    def __init__(self, openpose_path: Optional[str] = None, model_folder: Optional[str] = None):
        """
        키포인트 추출기 초기화 (pyopenpose Python API 사용)
        
        Args:
            openpose_path: OpenPose 빌드 경로 (예: "/path/to/openpose")
            model_folder: OpenPose 모델 폴더 경로 (예: "/path/to/openpose/models")
        """
        if not OPENPOSE_AVAILABLE:
            raise ImportError(
                "OpenPose가 설치되지 않았습니다. "
                "OpenPose를 설치하거나 pyopenpose를 사용하세요."
            )
        
        # OpenPose 파라미터 설정
        params = dict()
        
        if openpose_path:
            params["openpose_path"] = openpose_path
        
        if model_folder:
            params["model_folder"] = model_folder
        
        # 기본 설정
        params["model_pose"] = "BODY_25"  # BODY_25 모델 사용
        params["net_resolution"] = "-1x368"  # 네트워크 해상도
        params["output_resolution"] = "-1x-1"  # 출력 해상도
        params["number_people_max"] = 1  # 최대 1명만 감지
        params["disable_blending"] = False
        
        # OpenPose 객체 초기화
        try:
            self.op_wrapper = op.WrapperPython()
            self.op_wrapper.configure(params)
            self.op_wrapper.start()
        except Exception as e:
            raise RuntimeError(f"OpenPose 초기화 실패: {str(e)}")
    
    def extract_from_image(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        image_type: str = "full"
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        이미지에서 키포인트를 추출합니다.
        
        Args:
            image: 이미지 경로, PIL Image 객체, 또는 NumPy 배열
            image_type: 'full' 또는 'half' (추출할 키포인트 범위 결정)
        
        Returns:
            키포인트 딕셔너리: {키포인트명: (x, y, confidence), ...}
            사람이 감지되지 않으면 None 반환
        """
        # 이미지를 NumPy 배열로 변환
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image}")
        elif isinstance(image, Image.Image):
            # PIL Image를 NumPy 배열로 변환
            img_array = np.array(image)
            if img_array.shape[2] == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img_array = image.copy()
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # RGB인 경우 BGR로 변환
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")
        
        # OpenPose 처리
        datum = op.Datum()
        datum.cvInputData = img_array
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        # 키포인트 추출
        if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
            return None
        
        # 첫 번째 사람의 키포인트만 사용
        keypoints_array = datum.poseKeypoints[0]  # shape: (25, 3) - [x, y, confidence]
        
        # 키포인트 딕셔너리로 변환
        keypoints = {}
        
        # 키포인트 인덱스 선택
        if image_type == "half":
            indices = self.UPPER_BODY_INDICES
        else:  # full
            indices = range(len(self.KEYPOINT_NAMES))
        
        for idx in indices:
            if idx < len(keypoints_array):
                x, y, confidence = keypoints_array[idx]
                # confidence가 0이면 감지되지 않은 키포인트
                if confidence > 0:
                    keypoints[self.KEYPOINT_NAMES[idx]] = (float(x), float(y), float(confidence))
        
        return keypoints if keypoints else None
    
    def extract_from_keypoint_image(
        self,
        keypoint_image: Union[str, Path, Image.Image]
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        키포인트 이미지(스켈레톤 이미지)에서 키포인트 좌표를 추출합니다.
        
        주의: 이 메서드는 키포인트 이미지에서 직접 좌표를 추출하는 것이 아니라,
        키포인트 이미지를 일반 이미지로 처리하여 추출합니다.
        실제 키포인트 좌표가 필요한 경우 별도의 파싱 로직이 필요할 수 있습니다.
        
        Args:
            keypoint_image: 키포인트 이미지 경로 또는 PIL Image 객체
        
        Returns:
            키포인트 딕셔너리 또는 None
        """
        return self.extract_from_image(keypoint_image, image_type="full")
    
    def get_keypoint_list(
        self,
        keypoints: Dict[str, Tuple[float, float, float]],
        image_type: str = "full"
    ) -> List[Tuple[float, float, float]]:
        """
        키포인트 딕셔너리를 순서가 있는 리스트로 변환합니다.
        
        Args:
            keypoints: 키포인트 딕셔너리
            image_type: 'full' 또는 'half'
        
        Returns:
            [(x, y, confidence), ...] 형태의 리스트
        """
        if image_type == "half":
            indices = self.UPPER_BODY_INDICES
        else:
            indices = range(len(self.KEYPOINT_NAMES))
        
        keypoint_list = []
        for idx in indices:
            name = self.KEYPOINT_NAMES[idx]
            if name in keypoints:
                keypoint_list.append(keypoints[name])
            else:
                keypoint_list.append((0.0, 0.0, 0.0))  # 감지되지 않은 키포인트
        
        return keypoint_list
    
    def calculate_head_size(
        self,
        keypoints: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """
        머리 크기를 계산합니다 (PCKh 계산용).
        
        Args:
            keypoints: 키포인트 딕셔너리
        
        Returns:
            머리 크기 (픽셀 단위)
        """
        # OpenPose는 left_ear, right_ear 대신 다른 인덱스 사용
        # BODY_25에서는: left_ear=17, right_ear=18
        if "left_ear" in keypoints and "right_ear" in keypoints:
            left_ear = keypoints["left_ear"]
            right_ear = keypoints["right_ear"]
            head_size = np.sqrt(
                (left_ear[0] - right_ear[0]) ** 2 + 
                (left_ear[1] - right_ear[1]) ** 2
            )
            return head_size if head_size > 0 else 50.0
        
        # 귀가 없으면 눈 사이 거리 사용
        if "left_eye" in keypoints and "right_eye" in keypoints:
            left_eye = keypoints["left_eye"]
            right_eye = keypoints["right_eye"]
            head_size = np.sqrt(
                (left_eye[0] - right_eye[0]) ** 2 + 
                (left_eye[1] - right_eye[1]) ** 2
            )
            return head_size if head_size > 0 else 50.0
        
        # 기본값
        return 50.0


def get_openpose_executable_path(openpose_dir: Optional[str] = None) -> Optional[str]:
    """
    OpenPose 실행 파일 경로를 반환합니다.
    
    Args:
        openpose_dir: OpenPose 설치 디렉토리 (None이면 환경 변수에서 가져옴)
    
    Returns:
        실행 파일 경로 또는 None
    """
    if openpose_dir is None:
        openpose_dir = os.environ.get("OPENPOSE_DIR")
    
    if openpose_dir is None:
        return None
    
    openpose_dir = Path(openpose_dir)
    
    # 플랫폼별 실행 파일 이름
    if sys.platform == "win32":
        exe_name = "OpenPoseDemo.exe"
    elif sys.platform == "darwin":  # macOS
        exe_name = "openpose.bin"
    else:  # Linux
        exe_name = "openpose.bin"
    
    # 가능한 경로들
    possible_paths = [
        openpose_dir / "bin" / exe_name,
        openpose_dir / exe_name,
        openpose_dir / "build" / "examples" / "openpose" / exe_name,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def extract_keypoints_from_executable(
    image_path: Union[str, Path],
    model_name: str,
    prompt_version: str,
    openpose_dir: Optional[str] = None,
    output_base_dir: str = "extracted_keypoints"
) -> Optional[Path]:
    """
    OpenPose 실행 파일을 사용하여 이미지에서 키포인트를 추출하고 JSON으로 저장합니다.
    
    Args:
        image_path: 입력 이미지 경로
        model_name: 모델 이름 (예: 'nano_banana')
        prompt_version: 프롬프트 버전 (예: 'short', 'medium', 'long')
        openpose_dir: OpenPose 설치 디렉토리 (None이면 환경 변수 OPENPOSE_DIR 사용)
        output_base_dir: 출력 기본 디렉토리 (기본값: 'extracted_keypoints')
    
    Returns:
        생성된 JSON 파일 경로 또는 None (실패 시)
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # OpenPose 실행 파일 경로 확인
    openpose_exe = get_openpose_executable_path(openpose_dir)
    if openpose_exe is None:
        raise RuntimeError(
            "OpenPose 실행 파일을 찾을 수 없습니다. "
            "OPENPOSE_DIR 환경 변수를 설정하거나 openpose_dir 인자를 제공하세요."
        )
    
    # 출력 디렉토리 설정
    output_dir = Path(output_base_dir) / model_name / prompt_version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 임시 디렉토리 생성 (OpenPose JSON 출력용)
    temp_json_dir = output_dir / "_temp_json"
    temp_json_dir.mkdir(parents=True, exist_ok=True)
    
    # 출력 JSON 파일 경로
    output_json_path = output_dir / f"{image_path.stem}_keypoints.json"
    
    try:
        # OpenPose 명령어 구성
        cmd = [
            openpose_exe,
            "--image_path", str(image_path),
            "--write_json", str(temp_json_dir),
            "--display", "0",
            "--render_pose", "0",  # 렌더링 비활성화 (속도 향상)
        ]
        
        # 플랫폼별 추가 옵션
        if sys.platform == "win32":
            # Windows에서는 추가 옵션 없음
            pass
        else:
            # macOS/Linux에서는 모델 경로 지정 가능
            if openpose_dir:
                model_folder = Path(openpose_dir) / "models"
                if model_folder.exists():
                    cmd.extend(["--model_folder", str(model_folder)])
        
        # OpenPose 실행
        result = subprocess.run(
            cmd,
            cwd=openpose_dir if openpose_dir else None,
            check=True,
            capture_output=True,
            text=True
        )
        
        # 생성된 JSON 파일 찾기
        # OpenPose는 입력 파일명에 "_keypoints"를 붙여서 JSON 파일 생성
        expected_json_name = f"{image_path.stem}_keypoints.json"
        temp_json_file = temp_json_dir / expected_json_name
        
        if not temp_json_file.exists():
            # 다른 가능한 이름들 시도
            json_files = list(temp_json_dir.glob("*.json"))
            if json_files:
                temp_json_file = json_files[0]
            else:
                raise FileNotFoundError(
                    f"OpenPose가 JSON 파일을 생성하지 않았습니다. "
                    f"출력: {result.stdout}\n에러: {result.stderr}"
                )
        
        # 최종 출력 위치로 복사
        import shutil
        shutil.copy2(temp_json_file, output_json_path)
        
        # 임시 파일 정리
        temp_json_file.unlink()
        temp_json_dir.rmdir()
        
        return output_json_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ OpenPose 실행 실패: {e}")
        if e.stderr:
            print(f"에러 메시지: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ 키포인트 추출 중 오류 발생: {e}")
        return None


def load_keypoints_from_json(json_path: Union[str, Path]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """
    OpenPose JSON 파일에서 키포인트를 로드합니다.
    
    Args:
        json_path: OpenPose JSON 파일 경로
    
    Returns:
        키포인트 딕셔너리: {키포인트명: (x, y, confidence), ...} 또는 None
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")
    
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
    keypoint_names = KeypointExtractor.KEYPOINT_NAMES
    
    for idx, name in enumerate(keypoint_names):
        if idx < len(pose_keypoints):
            x, y, confidence = pose_keypoints[idx]
            if confidence > 0:
                keypoints[name] = (float(x), float(y), float(confidence))
    
    return keypoints if keypoints else None


def extract_keypoints(
    image_path: Union[str, Path],
    image_type: str = "full",
    openpose_path: Optional[str] = None,
    model_folder: Optional[str] = None
) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """
    편의 함수: 이미지에서 키포인트를 추출합니다 (pyopenpose Python API 사용).
    
    Args:
        image_path: 이미지 경로
        image_type: 'full' 또는 'half'
        openpose_path: OpenPose 빌드 경로 (선택사항)
        model_folder: OpenPose 모델 폴더 경로 (선택사항)
    
    Returns:
        키포인트 딕셔너리 또는 None
    """
    extractor = KeypointExtractor(openpose_path=openpose_path, model_folder=model_folder)
    return extractor.extract_from_image(image_path, image_type)
