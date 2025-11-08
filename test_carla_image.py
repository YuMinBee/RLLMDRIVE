"""
실제 이미지로 Vision-Language 모델 테스트 (BLIP-2)
"""

from blip2_model import BLIP2VisionLanguageModel
from PIL import Image


def test_with_real_image(image_path):
    """실제 이미지로 테스트"""
    
    print("=" * 60)
    print("🚗 Carla 이미지 Vision-Language 테스트 (BLIP-2)")
    print("=" * 60)
    
    # 1. 모델 로드
    print("\n1️⃣ BLIP-2 모델 로딩 중...")
    model = BLIP2VisionLanguageModel("Salesforce/blip2-opt-2.7b")
    print("✅ 모델 로딩 완료!\n")
    
    # 2. 이미지 로드
    print(f"2️⃣ 이미지 로드: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ 이미지 크기: {image.size}\n")
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return
    
    # 3. 이미지만으로 설명 생성
    print("3️⃣ 이미지 자동 설명 생성...")
    description = model.describe_image(image)
    print(f"💬 자동 설명: {description}\n")
    
    # 4. 다양한 프롬프트로 테스트
    print("4️⃣ 질문-응답 테스트\n")
    
    prompts = [
        "Question: Describe what you see in this driving scene. Answer:",
        "Question: What is the road condition? Answer:",
        "Question: Are there any vehicles or obstacles ahead? Answer:",
        "Question: What is the weather like? Answer:",
        "Question: Describe the buildings and surroundings. Answer:",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"{'='*60}")
        print(f"📝 질문 {i}: {prompt}")
        print("-" * 60)
        
        try:
            response = model.generate_response(
                image, 
                prompt, 
                max_length=80
            )
            print(f"💬 응답: {response}")
        except Exception as e:
            print(f"❌ 생성 실패: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ BLIP-2는 사전 학습된 모델이므로 의미 있는 답변을 생성합니다!")
    print("💡 자율주행에 활용 가능한 장면 이해 정보를 제공합니다.")
    print("=" * 60)


if __name__ == "__main__":
    # Carla 이미지 경로
    image_path = "/home/msp/Downloads/TalkMedia_i_4d41c420dc6a.png.png"
    
    test_with_real_image(image_path)
