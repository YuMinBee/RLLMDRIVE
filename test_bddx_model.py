"""
BDD-X 파인튜닝 LLaVA 모델 테스트
자율주행 특화 모델 vs 일반 모델 비교
"""

from llava_bddx_model import LLaVABDDXModel
from PIL import Image


def test_bddx_model(image_path):
    """BDD-X 파인튜닝 모델 테스트"""
    
    print("=" * 70)
    print("🚗 BDD-X 파인튜닝 LLaVA 모델 테스트")
    print("=" * 70)
    
    # 1. 모델 로드
    print("\n1️⃣ BDD-X 파인튜닝 모델 로딩 중...")
    model = LLaVABDDXModel()
    print("✅ 모델 로딩 완료!\n")
    
    # 2. 이미지 로드
    print(f"2️⃣ 이미지 로드: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ 이미지 크기: {image.size}\n")
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return
    
    # 3. 객체 인식 테스트
    print("3️⃣ 객체 인식 테스트...")
    
    detection_questions = [
        "Can you see lane markings? Describe them.",
        "Are there any buildings visible? Where are they?",
        "Can you see curbs or sidewalks?",
        "Is the vehicle hood visible in the image?",
        "Are there any street lights or traffic lights?",
        "What other vehicles are present? Describe their positions.",
    ]
    
    for i, question in enumerate(detection_questions, 1):
        print(f"\n   🔍 {i}. {question}")
        answer = model.generate_response(image, question, max_new_tokens=80)
        print(f"   ✓ {answer}")
    
    print()
    
    # 4. 주행 행동 결정
    print("4️⃣ 주행 행동 결정...")
    action_prompt = """Based on this driving scene, choose ONE action from the following options:
- Lane Keep (차선 유지)
- Lane Change Left (좌측 차선 변경)
- Lane Change Right (우측 차선 변경)
- Turn Left (좌회전)
- Turn Right (우회전)
- Stop (정지)
- Follow Front Vehicle (앞차 추종)

Which action should the autonomous vehicle take? Explain why."""
    
    action_decision = model.generate_response(image, action_prompt, max_new_tokens=150)
    print(f"🎯 행동 결정:\n{action_decision}\n")
    
    # 5. 안전성 판단
    print("5️⃣ 안전성 및 세부 질문\n")
    
    safety_questions = [
        "Is it safe to proceed? Why or why not?",
        "What is the current traffic situation?",
        "Are there any pedestrians or cyclists?",
        "What should be the appropriate speed for this situation?",
    ]
    
    for i, question in enumerate(safety_questions, 1):
        print(f"{'='*70}")
        print(f"❓ 질문 {i}: {question}")
        print("-" * 70)
        
        try:
            answer = model.generate_response(image, question, max_new_tokens=100)
            print(f"💬 응답: {answer}")
        except Exception as e:
            print(f"❌ 응답 생성 실패: {e}")
        
        print()
    
    print("=" * 70)
    print("✅ BDD-X 파인튜닝 모델 테스트 완료!")
    print("💡 자율주행 데이터로 학습되어 더 정확한 판단 기대")
    print("=" * 70)


if __name__ == "__main__":
    # Carla 이미지 경로
    image_path = "/home/msp/Downloads/test1.png"
    
    test_bddx_model(image_path)
