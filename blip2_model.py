"""
BLIP-2 기반 Vision-Language 모델
사전 학습된 모델로 실제 이미지 이해 가능
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image


class BLIP2VisionLanguageModel:
    """BLIP-2 기반 Vision-Language 모델"""
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        """
        Args:
            model_name: BLIP-2 모델 이름
                - "Salesforce/blip2-opt-2.7b" (2.7B, 권장)
                - "Salesforce/blip2-flan-t5-xl" (3B, 더 정확)
        """
        print(f"🔄 BLIP-2 모델 로딩 중: {model_name}")
        print("⏳ 처음 실행 시 모델 다운로드(~5GB)에 시간이 걸립니다...")
        
        # 프로세서 및 모델 로드
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        print(f"✅ 모델 로딩 완료! (Device: {self.device})")
    
    def generate_response(self, image, prompt, max_length=100):
        """
        이미지와 텍스트 프롬프트로 응답 생성
        
        Args:
            image: PIL Image 또는 이미지 경로
            prompt: 텍스트 질문/프롬프트
            max_length: 최대 생성 길이
        Returns:
            generated_text: 생성된 텍스트
        """
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 입력 준비
        inputs = self.processor(image, prompt, return_tensors="pt").to(
            self.device, 
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,  # Beam search로 더 나은 결과
                temperature=1.0,
                do_sample=False
            )
        
        # 디코딩
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text.strip()
    
    def describe_image(self, image):
        """이미지 설명 생성 (프롬프트 없이)"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        inputs = self.processor(image, return_tensors="pt").to(
            self.device,
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        return description.strip()
    
    def answer_question(self, image, question):
        """이미지 기반 질문 응답 (VQA)"""
        return self.generate_response(image, f"Question: {question} Answer:", max_length=50)


def main():
    """사용 예시"""
    
    print("=" * 70)
    print("BLIP-2 Vision-Language 모델 초기화")
    print("=" * 70)
    
    # 모델 초기화 (더 가벼운 버전 사용)
    # GPU 메모리가 충분하면 "Salesforce/blip2-opt-2.7b" 사용
    # 메모리 부족하면 더 작은 모델 사용
    try:
        model = BLIP2VisionLanguageModel("Salesforce/blip2-opt-2.7b")
    except Exception as e:
        print(f"⚠️  메모리 부족 가능성. 더 작은 모델 시도 중...")
        model = BLIP2VisionLanguageModel("Salesforce/blip2-flan-t5-xl")
    
    print("\n" + "=" * 70)
    print("✅ 모델 준비 완료!")
    print("=" * 70)
    print("\n사용 방법:")
    print("1. 이미지 설명:")
    print("   description = model.describe_image('image.jpg')")
    print("\n2. 질문 응답:")
    print("   answer = model.answer_question('image.jpg', 'What do you see?')")
    print("\n3. 커스텀 프롬프트:")
    print("   response = model.generate_response('image.jpg', 'Describe the scene:')")
    
    return model


if __name__ == "__main__":
    model = main()
