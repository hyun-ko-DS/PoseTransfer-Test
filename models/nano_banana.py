import mimetypes
import os
import io
import time
from pathlib import Path

from PIL import Image

from google import genai
from google.genai import types

from utils import img_to_bytes, save_binary_file


class NanoBanana:
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
        
        # 프롬프트가 비어있으면 기본값 사용
        if not prompt or not prompt.strip():
            prompt = "Apply the keypoint pose to the given image."
        
        # 환경 변수에서 API 키를 가져옵니다.
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. "
                ".env 파일을 생성하거나 환경 변수를 설정해주세요."
            )
        
        client = genai.Client(api_key=api_key)
        model = "gemini-3-pro-image-preview"

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=img_to_bytes(input_image),
                            mime_type="image/png"
                        )
                    ),
                    types.Part(
                        inline_data=types.Blob(
                            data=img_to_bytes(pose_image),
                            mime_type="image/png"
                        )
                    ),
                    types.Part(
                        text=prompt
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(image_size="1K"),
        )

        output_path = Path(output_path)
        saved_path = None
        
        # 재시도 로직 (최대 3번 시도)
        max_retries = 3
        retry_delay = 2  # 초
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # 코랩 코드와 동일한 구조로 실행
                last_chunk = None
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    last_chunk = chunk
                    if (
                        not chunk.candidates
                        or not chunk.candidates[0].content
                        or not chunk.candidates[0].content.parts
                    ):
                        continue

                    part = chunk.candidates[0].content.parts[0]
                    if part.inline_data and part.inline_data.data:
                        data_buffer = part.inline_data.data
                        file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                        saved_path = str(output_path).replace(".png", file_extension)
                        save_binary_file(saved_path, data_buffer)
                    else:
                        if hasattr(chunk, 'text') and chunk.text:
                            print(chunk.text)
                
                generation_time = time.time() - start_time
                
                # 입력/출력 토큰 계산
                input_tokens = 0
                output_tokens = 0
                if last_chunk and hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata:
                    input_tokens = last_chunk.usage_metadata.prompt_token_count
                    output_tokens = last_chunk.usage_metadata.candidates_token_count
                else:
                    token_count = client.models.count_tokens(model=model, contents=contents)
                    input_tokens = token_count.total_tokens
                    output_tokens = 0
                
                total_tokens = input_tokens + output_tokens
                print(f"   입력 토큰: {input_tokens:,}, 출력 토큰: {output_tokens:,}, 총 토큰: {total_tokens:,}")
                
                # 성공적으로 완료되면 반환 (시간과 토큰 정보 포함)
                result_path = saved_path if saved_path else str(output_path)
                return {
                    "saved_path": result_path,
                    "generation_time": generation_time,
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
            except Exception as e:
                error_str = str(e)
                # 500 에러나 재시도 가능한 에러인 경우
                if "500" in error_str or "INTERNAL" in error_str or "503" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)  # 지수 백오프
                        print(f"⚠️  API 에러 발생 (시도 {attempt + 1}/{max_retries}): {error_str}")
                        print(f"   {wait_time}초 후 재시도...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 마지막 시도 실패
                        raise Exception(f"API 호출 실패 (최대 재시도 횟수 초과): {error_str}")
                else:
                    # 재시도 불가능한 에러는 즉시 raise
                    raise
        
        # 모든 재시도 실패 시
        result_path = saved_path if saved_path else str(output_path)
        return {
            "saved_path": result_path,
            "generation_time": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }
