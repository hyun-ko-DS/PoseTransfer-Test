"""
키포인트 추출 모듈 (OpenPose 기반)

OpenPose를 사용하여 이미지에서 키포인트를 추출합니다.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2

try:
    import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("⚠️  OpenPose가 설치되지 않았습니다. OpenPose를 설치하거나 pyopenpose를 사용하세요.")


class KeypointExtractor:
    """
    OpenPose를 사용하여 이미지에서 키포인트를 추출하는 클래스
    """
    
    # OpenPose BODY_25 키포인트 인덱스 (25개 키포인트)
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
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
        키포인트 추출기 초기화
        
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


def extract_keypoints(
    image_path: Union[str, Path],
    image_type: str = "full",
    openpose_path: Optional[str] = None,
    model_folder: Optional[str] = None
) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """
    편의 함수: 이미지에서 키포인트를 추출합니다.
    
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
