# run v1 — Qwen3-8B QLoRA 파이프라인 검증 결과

- 실행: 2026-08-22 ~ 08-23
- 계획서: `PLAN.md` (파이프라인) / `PLAN-WEBUI.md` (웹 콘솔)
- run: `outputs/qwen3-8b-lora-v1`
- 결론: **파이프라인 검증 목표 달성. G5 만 실패했고, 이는 데이터 설계에 대한 발견이다.**

---

## 1. 게이트 판정

| 게이트 | 기준 | 실측 | 판정 |
|---|---|---|---|
| G1 완주 | OOM 없이, peak VRAM < 22GB | 57/57 스텝, **10.25GB** | ✅ |
| G2 수렴 | train loss 30% 이상 감소 | 2.7425 → 0.0566 (**97.9%**) | ✅ |
| G3 왕복 | 어댑터 저장 후 재로드 추론 | 분리 프로세스에서 성공 | ✅ |
| G4 반영 | 목표 행동이 출력에 나타남 | strict **base 0% → tuned 100%** | ✅ |
| G5 보존 | 베이스 능력 비붕괴 | epoch 1·3 **모두 실패** | ❌ |
| G6 재현 | config 만으로 재실행 | yaml 기반, seed 42 고정 | ✅ |

---

## 2. 실측 환경

계획서 작성 시점의 가정과 실제 설치 버전이 크게 달랐다.

| 항목 | 값 |
|---|---|
| 실행 환경 | WSL2 Ubuntu 22.04 (kernel 6.18 microsoft-standard) |
| Python | 3.11.16 (uv, 프로젝트 로컬 `.cache/uv-python`) |
| torch | 2.13.0+cu130 |
| transformers | **5.15.1** |
| trl | **1.10.0** |
| peft / bitsandbytes | 0.20.0 / 0.50.1 |
| GPU | RTX 3090, compute 8.6, bf16 지원, 25.8GB 인식 |

CUDA 패스스루는 WSL 에서 그대로 동작했다. 드라이버 추가 설치가 필요 없었다.

---

## 3. 학습 지표

```
총 57 step (epoch 당 19), warmup 2 step
train_runtime  428.6초 (7분 9초)
peak VRAM      10.25 GB
어댑터 크기     84 MB
```

| 구간 | train loss | eval loss |
|---|---|---|
| step 1 | 2.7425 | — |
| step 19 (epoch 1) | 0.4013 | **1.0446** ← 최저 |
| step 38 (epoch 2) | 0.0986 | 1.3813 ← 반등 |
| step 57 (epoch 3) | 0.0566 | 1.3825 |

**eval loss 는 epoch 1 에서 바닥을 찍고 올라갔다.** train loss 가 0.057 까지 내려간 건
304 샘플을 사실상 외운 상태다. 계획서 §06 회귀 체크리스트의 과적합 신호 그대로다.

VRAM 은 계획서 예상(11~14GB)보다 낮았다. `max_seq_length` 를 1024 로 두었지만
샘플이 89~109 토큰이라 한도에 한참 못 미쳤기 때문이다.

---

## 4. G4 — 형식 준수율 (통과)

학습에 한 번도 나오지 않은 7개 주제, 30개 프롬프트. greedy 디코딩.

| 지표 | base | tuned |
|---|---|---|
| strict | 0% | **100%** |
| lenient | 0% | **100%** |

같은 프롬프트에 대한 대비 (`블랙홀이 뭐야?`):

- **base** — 마크다운 헤더·이모지·굵은 글씨가 섞인 산문. 중간에 중국어 토큰(`甚至是`)이 섞였다.
- **tuned** — `{"answer": "...", "confidence": 0.69, "tags": ["astronomy", "physics"]}`

base 와 tuned 는 **같은 모델에서 어댑터만 토글**해 뽑았다 (PLAN-WEBUI.md §1.3 규칙 B).
실측 VRAM 6.44GB. 두 모델을 따로 올렸다면 11GB+ 였다.

---

## 5. G5 — 일반 능력 보존 (실패)

학습 도메인과 무관한 10개 질문. epoch 1·3 체크포인트 모두 실패했다.

### epoch 3 (최종 어댑터)

10개 중 9개가 JSON 으로만 답했고, **내용이 과제와 무관**했다.

| 질문 | 응답 | 문제 |
|---|---|---|
| 영어로 번역해줘 | "회의는 15시 30분에 시작됩니다." | 번역하지 않음 |
| 파이썬 중복 제거 **코드** | "집합은 순서가 없고..." | 코드 없이 설명만 |
| 생일 선물 3가지 추천 | 의미 없는 문장 | `tags` 가 `["answer","confidence","tags"]` |
| 4행시 써줘 | "가을은 여름의 뜨거움이..." | 시가 아니라 설명 |
| 문법 오류 고쳐줘 | `{"answer": "{"answer": ...` | 무한 반복 |

정상 응답은 3개(수열, 1~100 합, 재귀 함수).

### epoch 1 (checkpoint-19)

G4 는 동일하게 100%. 내용 품질은 **더 나았다** — 번역이 실제 영어로 나왔고
(`The meeting was postponed to 3 PM.`) 4행시도 시적인 문장이 나왔다.
그러나 무한 반복 붕괴가 **3건으로 오히려 늘었다.**

### greedy 아티팩트가 아님

무한 반복이 디코딩 방식 탓인지 확인하려고 샘플링으로 재생성했다.
**6회 중 5회 재현** — 모델 자체의 붕괴다.

---

## 6. 원인 — 데이터 설계

`PLAN.md` §04 에서 **"system 프롬프트로 규칙을 알려주지 않는다"**고 정했다.
G4 판정을 선명하게 만드는 데는 성공했지만(base 0% 가 그 증거), 대가가 있었다.

모델에게 **형식 모드와 일반 모드를 구분할 단서가 없다.** 게다가 304 샘플 전부가
"질문 → 짧은 사실 JSON" 한 종류다. 그래서 변환형 과제(번역·교정·재작성)를 만나면
맞는 연속을 찾지 못하고 `{"answer": "{"answer":` 루프에 빠진다.
실제로 붕괴한 3건이 전부 변환형 과제였다.

**epoch 를 줄여도 해결되지 않았다.** 계획서 §10 의 대응(epoch 하향)은
이 실패 모드에는 듣지 않는다. 원인이 학습량이 아니라 데이터 구성이기 때문이다.

---

## 7. 계획 가정과 달랐던 것들

검증 과정에서 계획서의 전제가 틀린 것으로 드러난 항목이다.

| 계획서 | 실제 |
|---|---|
| §3.2 "`<think>` 블록이 들어가면 안 된다" | `enable_thinking=False` 는 **빈 `<think></think>` 를 붙이는 것이 정상 표현**이다. 추론 프롬프트와 일치하므로 학습 데이터에 들어가는 게 맞다. 문제는 **내용이 있는** think 뿐이다. |
| §4.3 `assistant_only_loss` 로 loss 마스킹 | Qwen3 템플릿에 `{% generation %}` 이 없어 **마스크가 전부 0** 으로 나온다. 에러 없이 조용히 무효가 된다. → TRL prompt/completion 포맷 + `completion_only_loss` 로 전환 |
| §05 `warmup_ratio=0.03` | transformers 5 에서 제거됨. `warmup_steps` 로 총 스텝에서 환산 |
| §05 예상 VRAM 11~14GB | 실측 10.25GB |
| §02 Unsloth 미사용 | 유지. 대신 **gcc 가 없어** triton JIT 이 막혔다 (아래) |

---

## 8. 환경 구성에서 막혔던 지점

| 문제 | 해결 |
|---|---|
| WSL Ubuntu 에 gcc 없음, apt 는 sudo 필요 | PyPI `ziglang` 을 C 컴파일러로 사용. 툴체인도 프로젝트 안에 남는다 |
| triton 의 `-l:libcuda.so.1` 을 zig lld 가 못 찾음 | 래퍼에서 `-lcuda` 로 변환 + `.cache/lib` 에 `.so` 링크 생성 |
| Windows 콘솔 cp949 | subprocess 에 `PYTHONIOENCODING=utf-8` 강제. 안 하면 자식이 한글 print 하다 죽는다 |
| `runner.py` 가 Windows venv 경로 하드코딩 | 플랫폼별 인터프리터 선택. WSL 에서 `Exec format error` 로 학습이 시작조차 안 됐다 |
| 캐시가 AppData·`~/.cache` 로 흩어짐 | `app/env.py` 에서 10개 환경변수를 프로젝트 `.cache` 로 고정. **`HF_XET_CACHE` 는 `HF_HOME` 을 따라가지 않아** 따로 지정해야 한다 |

---

## 9. 다음 라운드 처방

G5 를 해결하려면 데이터를 다시 설계해야 한다. 학습량 조절로는 안 된다.

1. **system 프롬프트로 형식을 지시**하고, system 이 없는 샘플도 섞어 학습한다.
   형식 모드와 일반 모드를 구분할 단서를 준다.
2. **일반 instruction 데이터를 혼합**한다(replay). catastrophic forgetting 의 표준 처방이다.
3. **변환형 과제**(번역·교정·요약·재작성)를 데이터에 포함해 다양성을 확보한다.
   지금 붕괴하는 유형이 정확히 이 유형이다.
4. lr 하향, LoRA r 하향은 그 다음에 본다.

평가 쪽도 보완할 점이 있다. 지금 G5 는 정성 확인이라 "붕괴 여부"를 눈으로 센다.
무한 반복 탐지처럼 **자동 판정 가능한 지표**를 만들면 다음 라운드가 빨라진다.

---

## 10. 산출물

```
outputs/qwen3-8b-lora-v1/
├── adapter/                 최종(epoch 3) — 84MB
├── checkpoint-19/           epoch 1, eval loss 최저
├── checkpoint-38/           epoch 2
├── checkpoint-57/           epoch 3
├── config.yaml              CLI 재현용
├── metrics.jsonl            step 별 지표
└── status.json

reports/
├── eval_qwen3-8b-lora-v1_*.md|json              epoch 3 평가
└── eval_qwen3-8b-lora-v1_checkpoint-19_*.md|json  epoch 1 평가
```

CLI 재현:

```bash
cd /mnt/d/Work/Private/Fine-Tuning
PYTHONPATH=$PWD .venv-wsl/bin/python scripts/03_train.py \
  --config configs/qwen3_8b_qlora.yaml --run qwen3-8b-lora-v1
```
