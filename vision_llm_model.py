"""
경량화된 Vision-Language 모델
CLIP (Vision Encoder) + Projector + GPT-2 (LLM)
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image


class VisionProjector(nn.Module):
    """CLIP 임베딩을 LLM 입력 차원으로 변환하는 프로젝터"""
    
    def __init__(self, clip_dim=512, llm_dim=768, hidden_dim=1024):
        super().__init__()
        # 2-layer MLP로 더 나은 매핑
        self.projector = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim)
        )
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: CLIP 출력 [batch_size, clip_dim]
        Returns:
            llm_features: LLM 입력 가능한 형태 [batch_size, llm_dim]
        """
        return self.projector(vision_features)


class VisionLanguageModel(nn.Module):
    """Vision + Language 통합 모델"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", 
                 llm_model_name="gpt2"):
        super().__init__()
        
        # CLIP Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # LLM (GPT-2)
        self.llm = GPT2LMHeadModel.from_pretrained(llm_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projector
        clip_dim = self.vision_encoder.config.hidden_size  # 768 for base
        llm_dim = self.llm.config.n_embd  # 768 for gpt2
        self.projector = VisionProjector(clip_dim, llm_dim)
        
        print(f"✅ 모델 초기화 완료!")
        print(f"   - CLIP 차원: {clip_dim}")
        print(f"   - LLM 차원: {llm_dim}")
    
    def encode_image(self, image):
        """
        이미지를 CLIP으로 인코딩
        
        Args:
            image: PIL Image 또는 이미지 경로
        Returns:
            vision_tokens: LLM에 입력 가능한 비전 토큰 [1, llm_dim]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # CLIP 전처리
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # CLIP 인코딩
        with torch.no_grad():
            vision_outputs = self.vision_encoder(**inputs)
            # [CLS] 토큰의 임베딩 사용
            vision_features = vision_outputs.pooler_output  # [1, clip_dim]
        
        # Projector로 LLM 차원으로 변환
        vision_tokens = self.projector(vision_features)  # [1, llm_dim]
        
        return vision_tokens
    
    def generate_response(self, image, text_prompt, max_length=50):
        """
        이미지와 텍스트 프롬프트를 받아서 응답 생성
        
        Args:
            image: PIL Image 또는 이미지 경로
            text_prompt: 텍스트 질문/프롬프트
            max_length: 최대 생성 길이
        Returns:
            generated_text: 생성된 텍스트
        """
        # 1. 이미지 인코딩
        vision_tokens = self.encode_image(image)  # [1, llm_dim]
        
        # 2. 텍스트 토큰화
        text_inputs = self.tokenizer(
            text_prompt, 
            return_tensors="pt", 
            padding=True
        )
        text_input_ids = text_inputs.input_ids  # [1, seq_len]
        
        # 3. 텍스트를 임베딩으로 변환
        text_embeds = self.llm.transformer.wte(text_input_ids)  # [1, seq_len, llm_dim]
        
        # 4. 비전 토큰과 텍스트 임베딩 결합
        # vision_tokens를 [1, 1, llm_dim] 형태로 변환
        vision_tokens = vision_tokens.unsqueeze(1)  # [1, 1, llm_dim]
        
        # 결합: [이미지 토큰] + [텍스트 임베딩]
        combined_embeds = torch.cat([vision_tokens, text_embeds], dim=1)  # [1, 1+seq_len, llm_dim]
        
        # 5. LLM으로 생성
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            max_length=combined_embeds.shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 6. 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_vision_embeddings(self, image):
        """
        이미지의 임베딩만 추출 (다른 용도로 사용 가능)
        
        Args:
            image: PIL Image 또는 이미지 경로
        Returns:
            embeddings: 이미지 임베딩 [1, llm_dim]
        """
        return self.encode_image(image)


def main():
    """사용 예시"""
    
    print("=" * 50)
    print("Vision-Language 모델 초기화 중...")
    print("=" * 50)
    
    # 모델 초기화
    model = VisionLanguageModel()
    model.eval()  # 추론 모드
    
    print("\n" + "=" * 50)
    print("테스트 준비 완료!")
    print("=" * 50)
    print("\n사용 방법:")
    print("1. 이미지 인코딩:")
    print("   vision_tokens = model.encode_image('image.jpg')")
    print("\n2. 텍스트 생성:")
    print("   response = model.generate_response('image.jpg', 'Describe this image:')")
    print("\n3. 임베딩 추출:")
    print("   embeddings = model.get_vision_embeddings('image.jpg')")
    
    # 테스트 이미지가 있다면 주석 해제하고 사용
    # test_image_path = "test.jpg"
    # response = model.generate_response(test_image_path, "What is in this image?")
    # print(f"\n생성된 응답: {response}")
    
    return model


if __name__ == "__main__":
    model = main()
