"""
날씨 조건 명시 테스트
눈이 없다고 가정하고 행동 결정
"""

from llava_bddx_model import LLaVABDDXModel
from PIL import Image


def test_with_weather_override(image_path):
    """날씨 조건을 명시적으로 지정하여 테스트"""
    
    print("=" * 70)
    print("🌞 날씨 조건 명시 테스트 (맑은 날씨 가정)")
    print("=" * 70)
    
    # 모델 로드
    print("\n1️⃣ BDD-X 모델 로딩 중...")
    model = LLaVABDDXModel()
    print("✅ 모델 로딩 완료!\n")
    
    # 이미지 로드
    print(f"2️⃣ 이미지 로드: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"✅ 이미지 크기: {image.size}\n")
    
    # 날씨 재확인 프롬프트
    print("3️⃣ 날씨 재확인 (명확한 지시)")
    weather_prompt = """Look at this image carefully. 
The road surface appears BRIGHT because of strong SUNLIGHT, NOT because of snow.
This is a CLEAR, SUNNY day with DRY road conditions.
There is NO snow on the road.

Now, describe the actual weather condition in this image."""
    
    weather = model.generate_response(image, weather_prompt, max_new_tokens=100)
    print(f"☀️ 날씨 재확인:\n{weather}\n")
    
    # 행동 결정 (맑은 날씨 조건)
    print("4️⃣ 행동 결정 (맑은 날씨 가정)")
    action_prompt = """IMPORTANT: This is a CLEAR, SUNNY day. The road is DRY, NOT snowy.

Based on this driving scene with CLEAR weather and DRY road, choose ONE action:
- Lane Keep (차선 유지)
- Lane Change Left (좌측 차선 변경)  
- Lane Change Right (우측 차선 변경)
- Turn Left (좌회전)
- Turn Right (우회전)
- Stop (정지)
- Follow Front Vehicle (앞차 추종)

Which action should the vehicle take? Explain based on CLEAR weather conditions."""
    
    action_clear = model.generate_response(image, action_prompt, max_new_tokens=150)
    print(f"🎯 행동 (맑은 날씨):\n{action_clear}\n")
    
    # 비교: 눈 가정 행동 결정
    print("5️⃣ 행동 결정 (눈 덮인 날씨 가정 - 비교용)")
    action_prompt_snow = """Assume this road is covered with snow.

Based on this SNOWY driving scene, choose ONE action:
- Lane Keep (차선 유지)
- Lane Change Left (좌측 차선 변경)
- Lane Change Right (우측 차선 변경)
- Turn Left (좌회전)
- Turn Right (우회전)
- Stop (정지)
- Follow Front Vehicle (앞차 추종)

Which action should the vehicle take? Explain based on SNOWY conditions."""
    
    action_snow = model.generate_response(image, action_prompt_snow, max_new_tokens=150)
    print(f"❄️ 행동 (눈 덮인 날씨):\n{action_snow}\n")
    
    # 결과 비교
    print("=" * 70)
    print("📊 결과 비교")
    print("=" * 70)
    print(f"\n☀️ 맑은 날씨 가정:")
    print(f"   {action_clear[:100]}...")
    print(f"\n❄️ 눈 덮인 날씨 가정:")
    print(f"   {action_snow[:100]}...")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    image_path = "/home/msp/Downloads/test1.png"
    test_with_weather_override(image_path)
