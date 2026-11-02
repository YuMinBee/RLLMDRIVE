# RLLMDRIVE: CARLA 장면에서 Vision-Language Model의 주행 판단 검증

CARLA 시뮬레이터 이미지를 Vision-Language Model(VLM)에 넣어, VLM이 도로 상황을 얼마나 정확히
이해하고 주행 행동을 제안할 수 있는지 확인한 초기 탐색 프로젝트입니다 (2025.11).

> 이 탐색에서 확인한 VLM 판단 오류는 후속 프로젝트
> **[VLM-Guided Planner Intervention](https://github.com/YuMinBee/vlm-guided-planner-intervention)**
> (2026 MIDAS CDP 최우수상)의 핵심 설계, 즉 *VLM은 고수준 의도만 내고 검증을 통과한 판단만
> 주행에 반영한다*로 이어졌습니다.

## 비교한 모델

| 모델 | 파일 | 특징 |
|---|---|---|
| CLIP + GPT-2 (직접 구성) | `vision_llm_model.py` | CLIP ViT 인코더 + 2층 MLP projector + GPT-2. VLM 구조 이해용 |
| LLaVA-1.5-7B | `llava_model.py` | 범용 VLM |
| LLaVA BDD-X 파인튜닝 | `llava_bddx_model.py` | 주행 설명 데이터(BDD-X)로 파인튜닝된 자율주행 특화 모델 |

## 주요 발견

| 관찰 | 내용 | 시사점 |
|---|---|---|
| **환경 오인식** | 햇빛이 강한 맑은 도로를 "눈 덮인 도로"로 설명 | 사전학습 데이터 편향이 장면 이해에 그대로 드러남 |
| **프롬프트 보정 효과** | 날씨 조건을 프롬프트에 명시하자 "맑고 건조한 도로"로 정확히 인식 | 외부 맥락 정보가 VLM 판단을 크게 바꿈 |
| **위험한 행동 제안** | 직진·좌회전만 가능한 교차로에서 "우회전"을 추천 | 지도·경로 정보 없이 VLM 행동 제안을 그대로 쓰면 위험 |
| **객체 인식은 양호** | 차량·건물 감지와 일반 장면 설명은 비교적 정확 | 장면 의미 판단 자체는 활용 가치가 있음 |

**결론:** VLM은 장면의 의미를 읽는 데 강하지만, 주행 행동을 직접 맡기기에는 오판 위험이 큽니다.
따라서 VLM에는 경로·위치 정보를 함께 주고, 출력은 기존 주행 계획기가 검증한 뒤 반영해야 합니다.
이 결론이 후속 프로젝트의 비동기 의도 게이팅 구조의 출발점입니다.

## 실행

```bash
pip install -r requirements.txt
python test_carla_image.py        # LLaVA 기본 모델로 CARLA 이미지 설명
python test_bddx_model.py         # BDD-X 파인튜닝 모델 비교
python test_weather_override.py   # 날씨 조건 명시 프롬프트 실험
python test_vision_llm.py         # CLIP + GPT-2 구성 동작 확인
```

환경: Python 3.8, PyTorch 1.12.1 (CUDA 11.6), Transformers 4.46.3.
