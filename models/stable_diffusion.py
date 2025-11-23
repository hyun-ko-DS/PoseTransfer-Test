"""
Stable Diffusion 모델 클래스
"""

from pathlib import Path
from PIL import Image

# TODO: Stable Diffusion 구현 필요
class StableDiffusion:
    def __init__(self, prompt_file_path):
        self.prompt_file_path = prompt_file_path
        with open(self.prompt_file_path, "r") as f:
            self.prompt = f.read()
    
    def generate(self, input_image, pose_image, output_path, prompt=None):
        """
        이미지 생성 메서드
        
        Args:
            input_image: PIL Image 객체 (원본 이미지)
            pose_image: PIL Image 객체 (키포인트 이미지)
            output_path: 출력 파일 경로 (Path 객체 또는 문자열)
            prompt: 프롬프트 텍스트 (None이면 self.prompt 사용)
        
        Returns:
            생성된 이미지 파일 경로
        """
        if prompt is None:
            prompt = self.prompt
        
        # TODO: Stable Diffusion API 호출 구현
        raise NotImplementedError("StableDiffusion.generate() 메서드가 아직 구현되지 않았습니다.")

