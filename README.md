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
├── llava_model.py              # LLaVA-1.5-7B 기본 모델
├── llava_bddx_model.py         # BDD-X 파인튜닝 모델 (자율주행 특화)
├── vision_llm_model.py         # CLIP + GPT-2 커스텀 모델 (실험용)
├── test_vision_llm.py          # 더미 이미지 테스트
├── test_carla_image.py         # Carla 이미지 테스트 (기본 LLaVA)
├── test_bddx_model.py          # BDD-X 모델 테스트
├── test_weather_override.py    # 날씨 조건 명시 테스트
├── main.py                     # 메인 진입점 (향후 사용)
├── requirements.txt            # 패키지 의존성
├── git_push.sh                # Git 자동 푸시 스크립트
└── README.md                  # 프로젝트 문서
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

- **LLaVA-7B 기본 모델 테스트 완료**
- **LLaVA BDD-X 파인튜닝 모델 테스트 완료**
- **날씨 조건 명시 실험 완료** - 프롬프트 개선 효과 확인

### 🔍 발견된 문제점 (2025.11.08)

#### 1. 환경 인식 오류
- **문제**: 밝은 포장도로를 눈 덮인 도로로 오인식
  - 실제: 맑은 날씨, 햇빛이 강한 상황
  - 모델 판단: "The road is covered in snow"
- **원인**: LLaVA 학습 데이터에서 밝은 도로 → 눈으로 학습된 편향
- **영향**: 날씨/도로 상태 판단 부정확 → 자율주행 의사결정 오류 가능
- **✅ 해결 방법 발견**: 프롬프트에 날씨 조건 명시 시 정확하게 인식
  - 명시 전: "The road is covered in snow"
  - 명시 후: "Clear, sunny day with dry road conditions"

#### 2. 주행 방향 판단 오류
- **문제**: 우회전 불가능한 도로에서 우회전 추천
  - 실제: 직진, 좌회전만 가능
  - 모델 판단: "I would recommend turning right"
- **원인**: Carla 도로 구조 특성 학습 부족
- **영향**: **위험한 주행 판단** - 실제 자율주행 적용 시 사고 위험

#### 3. 기타 발견 사항
- ✅ 차량/건물 감지는 비교적 정확
- ✅ 일반적인 장면 설명은 합리적
- ❌ Carla 특화 정보(도로 표지, 차선, 가능한 행동) 인식 부족
- ❌ 목적지 정보 없이도 "방향 추천" - 근거 불명확

### 📊 테스트 결과 요약

**테스트 이미지**: Carla 시뮬레이션 스크린샷 (test1.png)
- 실제 상황: 맑은 날씨, 교차로, 직진/좌회전 가능
- 모델 인식 (기본): 눈 덮인 도로, 우회전 추천
- 모델 인식 (날씨 명시): 맑은 날씨, 직진 권장 ✅

**사용 모델:**
1. **LLaVA-1.5-7B (기본)**: 일반 Vision-Language 모델
2. **LLaVA-BDD-X (파인튜닝)**: 자율주행 특화 모델

**질문-응답 정확도 (BDD-X 모델)**:
| 질문 | 정확도 | 비고 |
|------|--------|------|
| 날씨 조건 | ⚠️ 조건부 | 명시 필요 |
| 차선 인식 | ✅ 정확 | 노란 차선 인식 |
| 건물 인식 | ✅ 정확 | 배경 건물 감지 |
| 연석/보도 | ✅ 정확 | 있음 |
| 차량 보닛 | ✅ 정확 | 보임 |
| 가로등/신호등 | ✅ 정확 | 있음 |
| 다른 차량 | ✅ 정확 | 트럭, 주차 차량 감지 |
| 주행 방향 | ⚠️ 날씨 의존 | 날씨에 따라 다름 |

**행동 결정 비교 (날씨 조건별)**:
| 날씨 조건 | 선택 행동 | 근거 |
|----------|----------|------|
| 맑은 날씨 (명시) | 직진 유지 | 도로 건조, 장애물 없음 |
| 눈 덮임 (가정) | 앞차 추종 | 가시성/접지력 감소 |

### 🎯 개선 방향

#### 단기 (즉시 가능)
1. **프롬프트 엔지니어링**
   - 더 명확한 질문으로 정확도 향상
   - Few-shot 예시 제공
   - 제약 조건 명시 ("물리적으로 가능한 방향만")

2. **경고 메시지 해결**
   - ✅ processor 설정 추가 (patch_size, vision_feature_select_strategy)
   - ⚠️ do_sample 파라미터 조정 필요

#### 중기 (데이터 준비 후)
3. **파인튜닝 (Fine-tuning)**
   - Carla 이미지 + 정확한 라벨 데이터셋 구축
   - LoRA/QLoRA로 효율적 학습
   - 목표: Carla 특화 환경 인식 능력 향상

4. **출력 구조화**
   - JSON 형식 강제
   - 가능/불가능 행동 명시
   - 신뢰도 점수 추가

#### 장기 (팀 협업)
5. **강화학습 연동**
   - 구조화된 출력 → RL 에이전트 입력
   - 보상 함수와 연계
   
6. **ROS 통합**
   - 실시간 이미지 처리
   - 주행 명령 생성

### 💡 핵심 인사이트

> **사전 학습된 Vision-Language 모델(LLaVA)도 Carla 환경에서는 환경 인식이 부정확함**
> 
> - 일반 이미지 이해 능력 ≠ 자율주행 특화 이해
> - **프롬프트 엔지니어링으로 부분 해결 가능**: 날씨 조건 명시 시 정확도 향상
> - **BDD-X 파인튜닝 모델**: 객체 인식 우수, 행동 결정 명확
> - 안전성 critical: 잘못된 판단이 사고로 직결

**실험 결과:**
1. ✅ **객체 인식**: 차선, 건물, 차량, 신호등 등 매우 정확
2. ✅ **행동 결정**: 7가지 행동 중 명확한 선택 (조건별 다름)
3. ⚠️ **날씨 인식**: 프롬프트에 명시 필요
4. ⚠️ **Carla 특화**: 여전히 파인튜닝 필요

### 🔬 연구 진행 계획

1. **현재 단계**: 베이스라인 모델 테스트 및 문제 파악 ✅
2. **다음 단계**: 
   - [ ] 다양한 Carla 시나리오 테스트 (10+ 이미지)
   - [ ] 프롬프트 최적화 실험
   - [ ] 데이터셋 수집 파이프라인 구축
3. **향후 단계**:
   - [ ] 파인튜닝 실험
   - [ ] 정량적 평가 지표 정의
   - [ ] 성능 비교 분석

### 📝 다음 단계

1. **즉시 실행 가능**
   - [x] LLaVA 기본 모델 테스트
   - [x] BDD-X 파인튜닝 모델 테스트
   - [x] 날씨 조건 명시 실험
   - [ ] 다양한 Carla 시나리오 테스트 (10+ 이미지)
   - [ ] 프롬프트 템플릿 최적화

2. **단기 목표**
   - [ ] 구조화된 출력 포맷 개발 (JSON)
   - [ ] 7가지 행동 결정 확률 분포 출력
   - [ ] 신뢰도 점수 추가
   
3. **중기 목표**
   - [ ] Carla 데이터 수집 파이프라인
   - [ ] LoRA/PEFT로 Carla 특화 파인튜닝
   - [ ] 정량적 평가 지표 정의

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

| 모델 | 크기 | 상태 | 용도 | 비고 |
|------|------|------|------|------|
| CLIP + GPT-2 | ~650MB | Projector 미학습 | 구조 학습용 | 실험용 |
| ~~BLIP-2 (opt-2.7b)~~ | ~~10GB~~ | ~~삭제~~ | - | LLaVA로 대체 |
| **LLaVA-1.5-7B** | ~13GB | **사전 학습 완료** | 기본 모델 | 일반적 성능 |
| **LLaVA-BDD-X** | ~4GB | **파인튜닝 완료** | **메인 모델** | 자율주행 특화 ⭐ |

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
