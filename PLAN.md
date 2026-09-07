# Qwen3-8B Fine-Tuning 파이프라인 검증 계획서

- 작성일: 2026-08-22
- 대상 모델: `Qwen/Qwen3-8B`
- 목적: **파이프라인 검증** — 데이터 준비 → 학습 → 저장 → 추론 → 평가까지 한 바퀴가 실제로 돌아가는지 확인
- 작업 경로: `D:\Work\Private\Fine-Tuning`

---

## 1. 목표와 성공 기준

이번 라운드의 목표는 "모델을 똑똑하게 만드는 것"이 아니라 **엔드투엔드 루프를 확보하는 것**이다.
성능 개선은 파이프라인이 검증된 다음 라운드의 과제로 분리한다.

### 성공 기준 (Definition of Done)

| # | 기준 | 검증 방법 |
|---|---|---|
| G1 | 3090 24GB에서 OOM 없이 학습 완주 | 1 epoch 완료, peak VRAM < 22GB |
| G2 | train loss가 유의미하게 하강 | 시작 대비 30% 이상 감소, 발산 없음 |
| G3 | 학습된 LoRA 어댑터를 저장·재로드해 추론 성공 | 별도 스크립트에서 로드 후 생성 확인 |
| G4 | 학습 목표 행동이 출력에 실제로 반영됨 | before/after 동일 프롬프트 20개 비교 |
| G5 | 베이스 능력이 붕괴하지 않음 | 일반 상식/한국어 프롬프트 10개 정성 확인 |
| G6 | 전 과정 재현 가능 | 스크립트 + config만으로 처음부터 재실행 가능 |

**실패해도 되는 것**: 절대 성능, 벤치마크 점수, 데이터 규모.
**실패하면 안 되는 것**: 재현성, 저장/로드, 채팅 템플릿 정합성.

---

## 2. 환경 구성

### 2.1 현재 환경

| 항목 | 값 | 비고 |
|---|---|---|
| GPU | RTX 3090 24GB (Ampere) | bf16 지원 O, FA2 지원 O(Linux) |
| OS | Windows 11 Pro (26100) | |
| Python | 3.14.3 (시스템) | **PyTorch 미지원 — 사용 불가** |
| 기타 | LM Studio 설치됨 | 최종 GGUF 검증에 활용 |

### 2.2 실행 환경 선택

> **권장: WSL2 (Ubuntu 22.04)**

| | WSL2 | Windows 네이티브 |
|---|---|---|
| torch + CUDA | O | O |
| bitsandbytes (4bit) | O 안정 | O (공식 휠 있음) |
| flash-attention-2 | O | X → `sdpa` 폴백 |
| Unsloth / triton | O | 불안정 |
| GPU 메모리 오버헤드 | 낮음 | WDDM 오버헤드 있음 |

Windows에서도 `transformers + peft + trl + bitsandbytes` 조합은 동작하므로,
WSL2 설치가 부담되면 네이티브로 시작하고 attention을 `sdpa`로 고정한다.
단 **Unsloth는 이번 라운드에서 쓰지 않는다** (Windows 이슈 + 검증 목적에 불필요한 변수).

### 2.3 패키지

Python **3.11** 가상환경을 별도로 만든다 (3.14는 사용하지 않음).

```
torch (CUDA 12.x)
transformers >= 4.51   # Qwen3 지원 최소 버전
peft
trl
datasets
accelerate
bitsandbytes
tensorboard
```

**검증 게이트**: 본 학습 전에 `torch.cuda.is_available()`, bf16 지원 여부,
4bit 로드 후 짧은 생성 1회를 먼저 통과시킨다. 여기서 막히면 학습으로 넘어가지 않는다.

---

## 3. 모델과 학습 방식

### 3.1 Qwen3-8B 선정 근거

- 한국어 성능이 동급 8B 중 우수 → 정성 평가 시 판단이 쉬움
- 4bit 양자화 시 가중치 약 5.5GB → 24GB에서 충분한 여유
- Chat template이 잘 정의되어 있어 SFT 포맷 실수를 줄일 수 있음

### 3.2 ⚠ Thinking 모드 주의 (이 계획서의 최대 함정)

Qwen3는 하이브리드 추론 모델로 `<think>...</think>` 블록을 출력하며,
chat template이 `enable_thinking` 인자를 받는다. 여기서 학습/추론 설정이 어긋나면
"학습은 됐는데 출력이 이상한" 전형적 실패가 발생한다.

**이번 라운드 방침: thinking OFF로 통일한다.**

- 데이터 생성 시: `tokenizer.apply_chat_template(..., enable_thinking=False)`
- 평가/추론 시: **동일하게** `enable_thinking=False`
- 학습 데이터의 assistant 응답에는 `<think>` 블록을 넣지 않는다

파이프라인 검증 단계에서 변수를 하나 줄이는 것이 목적이며,
thinking 학습은 다음 라운드로 분리한다.

### 3.3 QLoRA 설정

| 항목 | 값 | 근거 |
|---|---|---|
| 방식 | QLoRA (4bit NF4 + double quant) | 24GB 단일 GPU 제약 |
| compute dtype | bfloat16 | Ampere 지원, fp16보다 안정 |
| LoRA rank `r` | 16 | 검증 목적에 충분 |
| `lora_alpha` | 32 | alpha = 2r 관례 |
| `lora_dropout` | 0.05 | |
| target_modules | `q,k,v,o,gate,up,down_proj` | 전 linear 적용이 품질 안정적 |
| gradient checkpointing | ON | VRAM 확보 |
| attn_implementation | `sdpa` (WSL2면 `flash_attention_2`) | |

**예상 VRAM**: 가중치 5.5GB + 어댑터/옵티마이저 ~1GB + 활성값 4~7GB ≈ **11~14GB**.
여유가 크므로 OOM 시 seq_len → batch → checkpointing 순으로 조인다.

---

## 4. 데이터셋

### 4.1 규모와 성격

파이프라인 검증이므로 **작고, 효과가 눈에 띄고, 검증이 쉬운** 데이터를 쓴다.

- 규모: **300~500 샘플** (train 90% / eval 10%)
- 형태: 멀티턴 불필요, 싱글턴 instruction-response
- 최대 길이: 1024 토큰 (초과 샘플은 제외, truncate 금지)

### 4.2 데이터 선택지

| 안 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **A (권장)** | 공개 한국어 instruction 셋에서 300~500개 샘플링 | 즉시 시작 가능 | 학습 효과가 미묘해 G4 판정이 애매 |
| **B** | "모든 답변을 지정 JSON 스키마로" 같은 **형식 규칙**을 합성 데이터로 300개 생성 | 학습 여부가 **한눈에** 보임 → G4 판정 명확 | 데이터 생성 작업 필요 |
| **C** | 고정 페르소나(말투/이름/정체성) 200~300개 | 효과 확인 매우 쉬움 | 실제 태스크와 거리 있음 |

> **권장: B**. 파이프라인 검증에서 가장 중요한 건 "학습이 반영됐는지 명확히 판별 가능한가"이며,
> 형식 준수는 정답률로 **자동 채점**이 가능해 G4를 주관 없이 판정할 수 있다.
> (A는 loss만 보고 "된 것 같다"로 끝나기 쉬움)

### 4.3 포맷 규칙

- 저장 형식: JSONL, `{"messages": [{"role":"user",...},{"role":"assistant",...}]}`
- 학습 시 chat template 적용은 **토크나이저에 위임** (수동 문자열 조립 금지)
- **loss 마스킹**: user 구간은 loss에서 제외하고 assistant 응답에만 학습 (`completion_only`)
- 데이터 스모크 테스트: 학습 전 샘플 3개를 디코드해 특수 토큰/EOS/역할 경계를 눈으로 확인

---

## 5. 학습 설정

| 하이퍼파라미터 | 값 | 비고 |
|---|---|---|
| epochs | 3 | 소규모 데이터 기준 |
| per_device_batch_size | 2 | |
| gradient_accumulation_steps | 8 | 유효 배치 16 |
| learning_rate | 2e-4 | LoRA 표준 |
| scheduler | cosine | |
| warmup_ratio | 0.03 | |
| optimizer | `paged_adamw_8bit` | VRAM 절약 |
| max_seq_length | 1024 | |
| bf16 | True | |
| logging_steps | 5 | loss 곡선 해상도 확보 |
| eval_strategy | epoch | |
| save_strategy | epoch | 어댑터만 저장 |
| seed | 42 | 재현성 (G6) |

**예상 소요**: 500샘플 × 3 epoch ≈ 약 90 optimizer step → **20~40분** 내외.
1시간을 크게 넘기면 설정 오류를 의심한다.

---

## 6. 평가

### 6.1 정량

- **train/eval loss 곡선** (TensorBoard) — 하강 여부, eval loss 반등 = 과적합 신호
- **형식 준수율** (데이터 B 채택 시): 평가 프롬프트 30개 중 규칙을 지킨 비율.
  기대: base 낮음 → tuned 90%+. 이 수치가 G4의 판정 근거.

### 6.2 정성 — before/after A/B

동일 프롬프트를 base와 tuned에 각각 넣고 표로 비교한다. **생성 파라미터를 동일하게 고정**한다.

| 구분 | 프롬프트 수 | 확인 항목 |
|---|---|---|
| 학습 도메인 내 | 20 | 의도한 행동이 나타나는가 (G4) |
| 학습 도메인 외 | 10 | 일반 능력/한국어가 무너지지 않았는가 (G5) |

### 6.3 회귀 신호 체크리스트

- 같은 문장 반복 / 멈추지 않음 → EOS 학습 실패, 템플릿 불일치 의심
- 응답에 `<think>`나 특수 토큰 노출 → 3.2 설정 불일치
- 모든 질문에 학습 데이터 답변만 반복 → 과적합, epoch/lr 하향

---

## 7. 저장 및 배포 검증

1. **어댑터 저장** — `adapter_model.safetensors` (수십 MB)
2. **재로드 추론** — 학습 스크립트와 **분리된** 스크립트에서 base + 어댑터 로드 후 생성 (G3 판정)
3. **병합** — 어댑터를 base에 merge 후 fp16 전체 모델 저장 (약 16GB, 디스크 여유 확인)
4. **GGUF 변환 → LM Studio 로드** — llama.cpp로 Q4_K_M 변환 후 LM Studio에서 동일 프롬프트 확인

3~4단계는 **선택**이지만, 실사용까지의 경로를 한 번은 뚫어두는 것을 권장한다.
(LM Studio가 이미 설치되어 있어 검증 비용이 낮음)

---

## 8. 디렉토리 구조

```
D:\Work\Private\Fine-Tuning\
├── PLAN.md
├── requirements.txt
├── configs\
│   └── qwen3_8b_qlora.yaml      # 5장 하이퍼파라미터
├── data\
│   ├── raw\
│   ├── train.jsonl
│   └── eval.jsonl
├── scripts\
│   ├── 00_check_env.py          # 2.3 검증 게이트
│   ├── 01_build_dataset.py      # 데이터 생성/변환
│   ├── 02_inspect_data.py       # 4.3 스모크 테스트
│   ├── 03_train.py              # SFTTrainer
│   ├── 04_infer.py              # G3: 어댑터 재로드 추론
│   ├── 05_eval_ab.py            # 6.1/6.2 before/after
│   └── 06_merge_export.py       # 병합 + GGUF
├── outputs\
│   └── qwen3-8b-lora-v1\
└── reports\
    └── run_v1.md                # 결과 기록
```

---

## 9. 실행 단계

| 단계 | 작업 | 산출물 | 게이트 |
|---|---|---|---|
| S0 | Python 3.11 venv + 패키지 설치 | `requirements.txt` | `00_check_env.py` 통과 |
| S1 | 4bit 로드 + 짧은 생성 1회 | — | VRAM/생성 확인 |
| S2 | 데이터 생성 및 분할 | `train/eval.jsonl` | 스모크 테스트 통과 |
| S3 | **20샘플 × 1 epoch 드라이런** | — | 크래시 없이 완주 |
| S4 | 본 학습 | 어댑터 | G1, G2 |
| S5 | 재로드 추론 | — | G3 |
| S6 | before/after 평가 | 비교표 | G4, G5 |
| S7 | 병합 + GGUF + LM Studio (선택) | GGUF | — |
| S8 | 결과 정리 | `reports/run_v1.md` | G6 |

> **S3(드라이런)을 생략하지 않는다.** 소량으로 전 구간을 먼저 통과시키는 것이
> 파이프라인 검증에서 가장 비용 대비 효과가 큰 단계다.

---

## 10. 리스크 및 대응

| 리스크 | 징후 | 대응 |
|---|---|---|
| Python 3.14에 torch 설치 실패 | pip resolve 오류 | 3.11 venv 사용 (S0에서 선차단) |
| Windows에서 bitsandbytes/triton 오류 | import 에러 | WSL2 전환, Unsloth 미사용 |
| Chat template 불일치 | 출력에 특수 토큰/무한 생성 | 4.3 스모크 테스트, 학습·추론 템플릿 통일 |
| Thinking 모드 혼선 | `<think>` 노출 | 3.2대로 전 구간 `enable_thinking=False` |
| OOM | CUDA OOM | seq_len 512 → batch 1 → grad accum 증가 |
| 과적합 | eval loss 반등 | epoch 2로, lr 1e-4로 |
| 일반 능력 붕괴 | 도메인 외 답변 품질 저하 | LoRA r 하향, epoch 축소 |
| 디스크 부족 | 병합 시 실패 | 7장 3~4단계 생략 가능 |

---

## 11. 다음 라운드 (범위 밖)

파이프라인이 검증된 뒤에 다룬다.

- 실제 목표 데이터셋 구축 및 규모 확대
- Thinking 모드 학습 (`<think>` 포함 데이터)
- 하이퍼파라미터 탐색 (r, lr, epoch)
- 정량 벤치마크 도입
- DPO 등 선호 학습 단계
