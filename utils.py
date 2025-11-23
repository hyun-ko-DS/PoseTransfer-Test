import io
import os
from pathlib import Path
from PIL import Image, ImageOps


def save_binary_file(file_path, data):
    """바이너리 데이터를 파일로 저장합니다."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_path}")

def im_show(img_path):
    """이미지를 표시합니다. (matplotlib과 numpy가 필요합니다)"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    img = Image.open(img_path)
    img_np = np.array(img) ## 행렬로 변환된 이미지
    plt.imshow(img_np) ## 행렬 이미지를 다시 이미지로 변경해 디스플레이
    plt.axis('off')
    plt.show() ## 이미지 인터프린터에 출력
    print("📏 Image size:", img.size)        # (width, height)
    return img.size

def img_to_bytes(img, format="PNG", quality=95):
    """
    PIL.Image 객체를 바이트 데이터로 변환합니다.
    
    Args:
        img: PIL Image 객체
        format: 이미지 포맷 ("PNG", "JPEG" 등)
        quality: JPEG 품질 (1-100, PNG는 무시됨)
    
    Returns:
        바이트 데이터
    """
    buf = io.BytesIO()
    if format.upper() == "JPEG":
        # JPEG는 RGB 모드만 지원
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=format, optimize=True)
    return buf.getvalue()

def load_image(image_path):
    """
    이미지 파일을 PIL Image로 로드합니다.
    코랩 코드와 동일하게 EXIF 정보 처리 및 RGB 변환을 수행합니다.
    """
    from PIL import ImageOps
    img = Image.open(image_path)
    # 코랩 코드와 동일: EXIF transpose 후 RGB 변환
    img = ImageOps.exif_transpose(img.convert("RGB"))
    return img

def get_image_files(image_type, bg_type=None):
    """
    이미지 파일 경로 리스트를 반환합니다.
    
    Args:
        image_type: 'full', 'half', 'selfie'
        bg_type: 'bg', 'no_bg' (selfie의 경우 None)
    
    Returns:
        이미지 파일 경로 리스트
    """
    base_path = Path("data/images")
    
    if image_type == "selfie":
        image_dir = base_path / "selfie"
        return sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + 
                     list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG")))
    else:
        if bg_type == "bg":
            image_dir = base_path / image_type / "bg"
        elif bg_type == "no_bg":
            image_dir = base_path / image_type / "no_bg"
        else:
            raise ValueError(f"bg_type must be 'bg' or 'no_bg' for {image_type}")
        
        return sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + 
                     list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG")))

def get_keypoint_path(image_type, kp_num):
    """
    키포인트 이미지 경로를 반환합니다.
    
    Args:
        image_type: 'full', 'half' (selfie는 half 사용)
        kp_num: 1, 2, 3
    
    Returns:
        키포인트 이미지 경로
    """
    # selfie는 half 키포인트 사용
    kp_type = "half" if image_type == "selfie" else image_type
    return Path(f"data/keypoints/{kp_type}/kp_{kp_type}_{kp_num}.png")

def create_output_path(model_name, prompt_version, image_type, bg_type, image_name, kp_num):
    """
    출력 파일 경로를 생성합니다.
    
    Args:
        model_name: 'nano_banana', 'stable_diffusion', 'qwen_controlnet'
        prompt_version: 'short', 'medium', 'long'
        image_type: 'full', 'half', 'selfie'
        bg_type: 'bg', 'no_bg' (selfie의 경우 None이지만 경로에는 포함)
        image_name: 원본 이미지 파일명 (확장자 제외)
        kp_num: 1, 2, 3
    
    Returns:
        출력 파일 경로 (Path 객체)
    """
    # selfie의 경우 bg_type을 빈 문자열로 처리하거나 경로에서 제외
    if image_type == "selfie":
        output_dir = Path(f"results/{model_name}/{prompt_version}/{image_type}")
    else:
        output_dir = Path(f"results/{model_name}/{prompt_version}/{image_type}/{bg_type}")
    
    output_path = output_dir / f"{image_name}_kp_{kp_num}.png"
    return output_path