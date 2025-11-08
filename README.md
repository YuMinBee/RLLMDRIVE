# RLLMDRIVE - Vision-Language Model for Autonomous Driving

자율주행을 위한 Vision-Language 멀티모달 모델 프로젝트

## 📋 프로젝트 개요

Carla 시뮬레이터의 비전 정보를 LLM이 이해할 수 있도록 토큰화하여, 자율주행 의사결정에 활용하는 Vision-Language 모델 개발

## 🎯 목표

- Carla 시뮬레이션 이미지 → 텍스트 설명 변환
- 도로 상황, 장애물, 날씨 등 장면 이해
- 강화학습과 연동 가능한 구조화된 출력

## 🛠️ 개발 환경

### Python & 패키지
- **Python**: 3.8.10 (Carla, ROS 호환성)
- **PyTorch**: 1.12.1+cu116 (CUDA 11.6)
- **Transformers**: 4.46.3
- **가상환경**: venv (`/venv/`)

### 주요 라이브러리
```
torch==1.12.1
torchvision==0.13.1
transformers==4.46.3
pillow==10.4.0
accelerate==1.0.1
```

## 📂 프로젝트 구조

```
RLLMDRIVE/
├── blip2_model.py          # BLIP-2 Vision-Language 모델 (메인)
├── vision_llm_model.py     # CLIP + GPT-2 커스텀 모델 (실험용)
├── test_vision_llm.py      # 더미 이미지 테스트
├── test_carla_image.py     # Carla 이미지 테스트
├── main.py                 # 메인 진입점 (향후 사용)
├── requirements.txt        # 패키지 의존성
├── git_push.sh            # Git 자동 푸시 스크립트
└── README.md              # 프로젝트 문서
```

## 🚀 진행 상황

### ✅ 완료된 작업

1. **개발 환경 구축**
   - Python 3.8 가상환경 생성
   - PyTorch 1.12.1 설치 (Python 3.8 호환)
   - 의존성 문제 해결 (sympy, typing-extensions 등)

2. **모델 구현**
   - **CLIP + GPT-2 커스텀 모델** (`vision_llm_model.py`)
     - Vision Encoder: CLIP ViT-Base
     - Projector: 2-layer MLP (CLIP → GPT-2 차원 변환)
     - LLM: GPT-2
     - 구조 이해용, Projector 미학습 상태
   
   - **BLIP-2 모델** (`blip2_model.py`) ⭐ **현재 메인**
     - 사전 학습된 Vision-Language 모델
     - Salesforce/blip2-opt-2.7b (~10GB)
     - 실제 이미지 이해 가능
     - GPU 자동 감지 (CUDA/CPU)

3. **테스트 환경**
   - 더미 이미지 생성 및 테스트
   - Carla 이미지 테스트 스크립트
   - 다양한 프롬프트 실험 준비

4. **Git 버전 관리**
   - GitHub 저장소 연동 완료
   - 자동 푸시 스크립트 (`./git_push.sh`)

### 🔄 진행 중

- **BLIP-2 모델 다운로드** (~10GB, 인터넷 속도에 따라 시간 소요)
- Carla 이미지로 첫 테스트 대기 중

### 📝 다음 단계

1. **즉시 실행 가능**
   - [ ] BLIP-2 모델 테스트 (Carla 이미지)
   - [ ] Zero-shot 프롬프트 최적화
   - [ ] 다양한 주행 시나리오 테스트

2. **단기 목표**
   - [ ] 모델 구조 직접 수정
     - Vision Encoder 교체 실험
     - Q-Former 커스터마이징
     - Projector 레이어 추가
   
3. **중기 목표**
   - [ ] 파인튜닝 준비
     - 공개 자율주행 데이터셋 실험 (BDD100K, KITTI)
     - LoRA/PEFT 환경 구축
     - Carla 데이터 수집 파이프라인
   - [ ] 모델 파인튜닝 (Carla 특화)

4. **장기 목표**
   - [ ] 출력 구조화 (팀원과 협의)
   - [ ] 강화학습 에이전트 연동
   - [ ] ROS 통합

## 🎮 사용 방법

### 설치
```bash
# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
```

### 실행
```bash
# BLIP-2 모델 테스트 (Carla 이미지)
python test_carla_image.py

# 더미 이미지 테스트
python test_vision_llm.py

# BLIP-2 모델 직접 사용
python blip2_model.py
```

### Git 푸시
```bash
# 변경사항 자동 커밋 & 푸시
./git_push.sh "커밋 메시지"
```

## 📊 모델 비교

| 모델 | 크기 | 상태 | 용도 |
|------|------|------|------|
| CLIP + GPT-2 | ~650MB | Projector 미학습 | 구조 학습용 |
| BLIP-2 (opt-2.7b) | ~10GB | 사전 학습 완료 | 메인 모델 |

## 🔧 기술 스택

- **Vision Encoder**: CLIP ViT / BLIP-2 ViT-L
- **Language Model**: GPT-2 / OPT-2.7B
- **Framework**: PyTorch, Transformers
- **Simulation**: Carla (예정)
- **Integration**: ROS (예정)

## 💡 핵심 아이디어

1. **경량화 우선**: Python 3.8 호환, 최소 의존성
2. **모듈화**: Vision/Language 컴포넌트 분리
3. **점진적 개선**: 
   - Zero-shot → Prompt Engineering → Fine-tuning
4. **실험 중심**: 구조 수정 및 다양한 시도

## 🐛 해결된 문제

### Python 버전 충돌
- **문제**: 최신 PyTorch(2.4.1)의 의존성이 Python 3.9+ 요구
- **해결**: PyTorch 1.12.1로 다운그레이드 (Python 3.8 호환)
- **교훈**: Carla/ROS 환경에서는 패키지 버전 호환성 중요

### 의존성 문제
- sympy, typing-extensions, filelock 등 버전 충돌
- PyTorch 버전 조정으로 일괄 해결

## 📚 참고 자료

- [BLIP-2 논문](https://arxiv.org/abs/2301.12597)
- [Hugging Face BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)
- [Carla 시뮬레이터](https://carla.org/)

## 👥 협업

- 출력 구조화: 팀원과 인터페이스 협의 필요
- 강화학습 연동: 추후 통합 예정

## 📝 메모

- BLIP-2 모델 다운로드 시간: 인터넷 속도에 따라 15분-1시간
- GPU 메모리: 최소 8GB 권장 (float16 사용)
- CPU 모드도 지원하지만 느림

---

**Last Updated**: 2025년 11월 8일  
**Status**: 🟡 개발 진행 중
