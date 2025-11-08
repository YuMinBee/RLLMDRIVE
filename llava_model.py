"""
LLaVA-7B 기반 Vision-Language 모델
이미지 정보를 최대한 보존하여 LLM에 전달
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image


class LLaVAVisionLanguageModel:
    """LLaVA 기반 Vision-Language 모델"""
    
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        """
        Args:
            model_name: LLaVA 모델 이름
                - "llava-hf/llava-1.5-7b-hf" (7B, 권장)
                - "llava-hf/llava-1.5-13b-hf" (13B, 더 정확)
        """
        print(f"🔄 LLaVA 모델 로딩 중: {model_name}")
        print("⏳ 처음 실행 시 모델 다운로드(~13GB)에 시간이 걸립니다...")
        
        # 프로세서 및 모델 로드
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 미래 호환성을 위한 processor 설정 (transformers v4.47+)
        self.processor.patch_size = 14  # ViT-L의 patch size
        self.processor.vision_feature_select_strategy = 'default'  # LLaVA 기본 전략
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        print(f"✅ 모델 로딩 완료! (Device: {self.device})")
        print(f"   - Vision Encoder: CLIP ViT-L")
        print(f"   - LLM: Vicuna-7B")
        print(f"   - 이미지 토큰: 256개 (BLIP-2보다 8배 많음)")
    
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
        ).to(self.device, torch.float16 if torch.cuda.is_available() else torch.float32)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature=0.2,
                # top_p=0.9
            )
        
        # 디코딩 (프롬프트 제외)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.processor.decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def describe_image(self, image, detail_level="detailed"):
        """
        이미지 설명 생성
        
        Args:
            image: PIL Image 또는 이미지 경로
            detail_level: "brief" (간단) 또는 "detailed" (상세)
        """
        if detail_level == "brief":
            prompt = "Briefly describe this image."
        else:
            prompt = "Describe this image in detail."
        
        return self.generate_response(image, prompt, max_new_tokens=150)
    
    def answer_question(self, image, question):
        """이미지 기반 질문 응답 (VQA)"""
        return self.generate_response(image, question, max_new_tokens=100)
    
    def analyze_driving_scene(self, image):
        """자율주행 장면 분석 (특화 프롬프트)"""
        prompt = """Analyze this driving scene and describe:
1. Road conditions (surface, weather)
2. Visible obstacles or vehicles
3. Traffic signs or signals
4. Safe driving recommendations"""
        
        return self.generate_response(image, prompt, max_new_tokens=250)


def main():
    """사용 예시"""
    
    print("=" * 70)
    print("LLaVA-7B Vision-Language 모델 초기화")
    print("=" * 70)
    
    # 모델 초기화
    model = LLaVAVisionLanguageModel("llava-hf/llava-1.5-7b-hf")
    
    print("\n" + "=" * 70)
    print("✅ 모델 준비 완료!")
    print("=" * 70)
    print("\n사용 방법:")
    print("1. 이미지 설명:")
    print("   description = model.describe_image('image.jpg')")
    print("\n2. 질문 응답:")
    print("   answer = model.answer_question('image.jpg', 'What do you see?')")
    print("\n3. 자율주행 장면 분석:")
    print("   analysis = model.analyze_driving_scene('carla_image.jpg')")
    print("\n4. 커스텀 프롬프트:")
    print("   response = model.generate_response('image.jpg', 'Your prompt')")
    
    return model


if __name__ == "__main__":
    model = main()
