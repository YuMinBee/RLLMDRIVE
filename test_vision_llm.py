"""
Vision-Language 모델 테스트 스크립트
"""

import torch
from vision_llm_model import VisionLanguageModel
from PIL import Image
import numpy as np


def create_dummy_image():
    """테스트용 더미 이미지 생성"""
    # 간단한 그라디언트 이미지 생성
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 빨간색 그라디언트
    for i in range(224):
        img_array[i, :, 0] = int(255 * i / 224)
    
    # 초록색 사각형
    img_array[50:150, 50:150, 1] = 255
    
    return Image.fromarray(img_array)


def test_model():
    """모델 테스트"""
    
    print("🚀 Vision-Language 모델 테스트 시작\n")
    
    # 1. 모델 초기화
    print("1️⃣ 모델 로딩 중...")
    model = VisionLanguageModel()
    model.eval()
    print("   ✅ 모델 로딩 완료!\n")
    
    # 2. 더미 이미지 생성
    print("2️⃣ 테스트 이미지 생성 중...")
    test_image = create_dummy_image()
    test_image.save("test_image.png")
    print("   ✅ 테스트 이미지 저장: test_image.png\n")
    
    # 3. 이미지 인코딩 테스트
    print("3️⃣ 이미지 인코딩 테스트...")
    vision_tokens = model.encode_image(test_image)
    print(f"   ✅ 비전 토큰 shape: {vision_tokens.shape}")
    print(f"   ✅ 비전 토큰 통계: mean={vision_tokens.mean():.4f}, std={vision_tokens.std():.4f}\n")
    
    # 4. 텍스트 생성 테스트
    print("4️⃣ 텍스트 생성 테스트...")
    prompts = [
        "Describe this image:",
        "What colors do you see?",
        "What objects are in this image?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   📝 프롬프트 {i}: '{prompt}'")
        response = model.generate_response(test_image, prompt, max_length=30)
        print(f"   💬 응답: {response}")
    
    print("\n" + "=" * 50)
    print("✅ 모든 테스트 완료!")
    print("=" * 50)
    print("\n다음 단계:")
    print("1. 실제 이미지로 테스트")
    print("2. Projector를 학습 데이터로 파인튜닝")
    print("3. 더 큰 LLM으로 교체 (phi-2, llama 등)")
    print("4. 멀티 이미지 처리 기능 추가")


if __name__ == "__main__":
    test_model()
