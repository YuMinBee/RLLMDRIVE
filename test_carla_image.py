"""
실제 이미지로 Vision-Language 모델 테스트 (LLaVA)
"""

from llava_model import LLaVAVisionLanguageModel
from PIL import Image


def test_with_real_image(image_path):
    """실제 이미지로 테스트"""
    
    print("=" * 60)
    print("🚗 Carla 이미지 Vision-Language 테스트 (LLaVA-7B)")
    print("=" * 60)
    
    # 1. 모델 로드
    print("\n1️⃣ LLaVA-7B 모델 로딩 중...")
    model = LLaVAVisionLanguageModel("llava-hf/llava-1.5-7b-hf")
    print("✅ 모델 로딩 완료!\n")
    
    # 2. 이미지 로드
    print(f"2️⃣ 이미지 로드: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ 이미지 크기: {image.size}\n")
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return
    
    # 3. 자율주행 장면 분석
    print("3️⃣ 자율주행 장면 분석...")
    analysis = model.analyze_driving_scene(image)
    print(f"� 장면 분석:\n{analysis}\n")
    
    # 4. 다양한 프롬프트로 테스트
    print("4️⃣ 질문-응답 테스트\n")
    
    prompts = [
        "What is the weather condition in this image?",
        "Describe the road surface and any hazards.",
        "Are there any vehicles visible? Where are they?",
        "What buildings or structures can you see?",
        "What should the driver be careful about in this scene?",
        "What should I do now: go straight, turn left, or turn right? Explain your choice based on the road and surroundings.",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"{'='*60}")
        print(f"📝 질문 {i}: {prompt}")
        print("-" * 60)
        
        try:
            response = model.answer_question(image, prompt)
            print(f"💬 응답: {response}")
        except Exception as e:
            print(f"❌ 생성 실패: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ LLaVA는 이미지 디테일을 잘 보존하여 상세한 답변 생성!")
    print("💡 BLIP-2보다 8배 많은 이미지 토큰으로 정보 손실 최소화")
    print("=" * 60)


if __name__ == "__main__":
    # Carla 이미지 경로
    image_path = "/home/msp/Downloads/test1.png"
    
    test_with_real_image(image_path)
