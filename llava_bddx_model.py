"""
LLaVA BDD-X 파인튜닝 모델
자율주행 특화 - Berkeley DeepDrive 데이터셋으로 학습
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image


class LLaVABDDXModel:
    """BDD-X 데이터셋으로 파인튜닝된 LLaVA 모델"""
    
    def __init__(self, 
                 base_model="llava-hf/llava-1.5-7b-hf",
                 adapter_model="Salmamoori/llava-bddx-finetuned"):
        """
        Args:
            base_model: 베이스 LLaVA 모델
            adapter_model: BDD-X 파인튜닝 어댑터
        """
        print(f"🔄 BDD-X 파인튜닝 LLaVA 모델 로딩 중...")
        print(f"   - Base: {base_model}")
        print(f"   - Adapter: {adapter_model}")
        print("⏳ 4-bit 양자화 모델 로딩 중...")
        
        # 4-bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(base_model)
        
        # 미래 호환성을 위한 processor 설정
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = 'default'
        
        # 베이스 모델 로드 (4-bit)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # LoRA 어댑터 로드
        print("📦 BDD-X 어댑터 로딩 중...")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_model,
            device_map="auto"
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"✅ 모델 로딩 완료! (Device: {self.device})")
        print(f"   - Vision Encoder: CLIP ViT-L")
        print(f"   - LLM: Vicuna-7B (4-bit)")
        print(f"   - 특화: 자율주행 (BDD-X 데이터셋)")
    
    def generate_response(self, image, prompt, max_new_tokens=200):
        """
        이미지와 텍스트 프롬프트로 응답 생성
        
        Args:
            image: PIL Image 또는 이미지 경로
            prompt: 텍스트 질문/프롬프트
            max_new_tokens: 최대 생성 토큰 수
        Returns:
            generated_text: 생성된 텍스트
        """
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # LLaVA 프롬프트 형식
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 프롬프트 템플릿 적용
        prompt_text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        # 입력 준비
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device, torch.float16)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9
            )
        
        # 디코딩 (프롬프트 제외)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.processor.decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def describe_driving_scene(self, image):
        """자율주행 장면 설명"""
        prompt = "Describe this driving scene in detail."
        return self.generate_response(image, prompt, max_new_tokens=150)
    
    def analyze_road_conditions(self, image):
        """도로 상황 분석"""
        prompt = "What are the road conditions and weather in this image?"
        return self.generate_response(image, prompt, max_new_tokens=100)
    
    def detect_obstacles(self, image):
        """장애물 감지"""
        prompt = "Are there any vehicles, pedestrians, or obstacles? Where are they located?"
        return self.generate_response(image, prompt, max_new_tokens=100)
    
    def recommend_action(self, image):
        """주행 행동 추천"""
        prompt = "Based on this driving scene, what should the driver do? Should they go straight, turn left, turn right, or stop?"
        return self.generate_response(image, prompt, max_new_tokens=150)


def main():
    """사용 예시"""
    
    print("=" * 70)
    print("LLaVA BDD-X 파인튜닝 모델 초기화")
    print("=" * 70)
    
    # 모델 초기화
    model = LLaVABDDXModel()
    
    print("\n" + "=" * 70)
    print("✅ 모델 준비 완료!")
    print("=" * 70)
    print("\n특징:")
    print("- BDD-X 데이터셋으로 파인튜닝")
    print("- 자율주행 장면 이해 특화")
    print("- 4-bit 양자화로 메모리 효율적")
    print("\n사용 방법:")
    print("1. 장면 설명:")
    print("   description = model.describe_driving_scene('image.jpg')")
    print("\n2. 도로 상황:")
    print("   conditions = model.analyze_road_conditions('image.jpg')")
    print("\n3. 장애물 감지:")
    print("   obstacles = model.detect_obstacles('image.jpg')")
    print("\n4. 행동 추천:")
    print("   action = model.recommend_action('image.jpg')")
    
    return model


if __name__ == "__main__":
    model = main()
