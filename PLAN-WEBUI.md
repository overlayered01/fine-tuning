# Web UI 작업 계획 — Gradio 학습 콘솔

- 작성일: 2026-08-22
- 상위 문서: `PLAN.md` (Qwen3-8B 파이프라인 검증 계획)
- 범위: 웹에서 **학습 실행 + 실시간 모니터링 + before/after 평가**
- 스택: **Gradio** (파이썬 단일 앱, 학습 venv 공용)

---

## 0. 이 문서의 핵심 원칙

> **UI는 CLI를 감싸는 얇은 레이어다.**

웹에서 시작한 학습도 `configs/*.yaml` + `scripts/03_train.py`만으로 동일하게 재현되어야 한다.
UI에만 존재하는 학습 로직을 만들면 `PLAN.md`의 **G6(재현성)**이 즉시 깨진다.
UI는 config를 만들고, 프로세스를 띄우고, 상태 파일을 읽어 그릴 뿐이다.

---

## 1. 아키텍처 — 가장 중요한 결정 3가지

### 1.1 학습은 Gradio 프로세스 안에서 돌리지 않는다

학습을 UI 프로세스 내부(같은 스레드/같은 CUDA 컨텍스트)에서 실행하면 세 가지가 한꺼번에 깨진다.

| 문제 | 결과 |
|---|---|
| UI 블로킹 | 학습 중 버튼·탭이 전부 먹통 |
| 중단 불가 | 실행 중인 `Trainer.train()`을 밖에서 안전하게 끊을 수 없음 |
| 크래시 전파 | 학습 OOM → 웹 서버까지 동반 사망, 로그도 유실 |

**결론: 학습은 `subprocess`로 분리한다.**

```
[브라우저] ──HTTP──> [Gradio app.py]
                          │  spawn (subprocess)
                          ▼
                    [03_train.py]  ← 실제 GPU 학습
                          │  write
                          ▼
                 status.json / metrics.jsonl / train.log
                          ▲
                          └── poll (gr.Timer, 1~2초) ── [Gradio app.py]
```

부수 효과로 얻는 것: **브라우저를 닫아도 학습은 계속되고, 다시 열면 상태가 복구된다.**
UI가 죽어도 학습은 살아 있다. 파일 기반이라 이게 공짜로 따라온다.

### 1.2 프로세스 간 통신은 파일로 한다

소켓·큐 대신 파일 3개면 충분하고, 훨씬 튼튼하다.

| 파일 | 쓰는 쪽 | 내용 | 갱신 |
|---|---|---|---|
| `status.json` | 학습 | `state`(idle/running/stopping/done/failed), `step`, `max_steps`, `started_at`, `error` | step마다 |
| `metrics.jsonl` | 학습 | `{step, loss, eval_loss, lr, vram_gb}` append-only | `logging_steps`마다 |
| `train.log` | 학습 | stdout/stderr 전체 | 실시간 |

- `status.json`은 **임시 파일에 쓰고 rename**(원자적 교체)한다. 그냥 덮어쓰면 UI가 반쪽짜리 JSON을 읽고 깨진다.
- `metrics.jsonl`은 append-only라 락이 필요 없다. UI는 읽은 offset만 기억하고 이어 읽는다.
- 구현부: 학습 쪽은 **HF `TrainerCallback`**의 `on_log` / `on_step_end`에 붙인다.

### 1.3 VRAM은 하나뿐이다 — 학습과 평가가 충돌한다

24GB 안에서 학습이 12~14GB를 쓰는 동안 평가 탭이 모델을 또 올리면 그대로 OOM이다.
두 가지 규칙으로 막는다.

**규칙 A — 학습 중 평가 탭 잠금.**
`status.state == running`이면 평가 탭의 실행 버튼을 비활성화하고 안내 문구를 띄운다.

**규칙 B — base와 tuned를 위해 모델을 두 번 올리지 않는다.**

이게 이번 설계에서 가장 이득이 큰 지점이다. PEFT 모델은 어댑터를 런타임에 끌 수 있다.

```python
# tuned 출력
out_tuned = model.generate(...)

# base 출력 — 같은 메모리, 어댑터만 비활성화
with model.disable_adapter():
    out_base = model.generate(...)
```

base 8B + tuned 8B를 따로 올리면 4bit라도 11GB+지만, 이 방식이면 **5.5GB 하나로 끝난다.**
before/after 비교가 목적인 §06 평가와 정확히 맞아떨어진다.

---

## 2. 화면 구성

### Tab 1 — 학습

| 영역 | 내용 |
|---|---|
| 설정 | base model, dataset 경로, r / alpha / lr / epochs / batch / grad_accum / max_seq_len / seed |
| 프리셋 | `PLAN.md` §05 기본값 1클릭 로드, **드라이런 프리셋**(20샘플 × 1epoch) 별도 버튼 |
| 실행 | `[학습 시작]` `[중단]` — 상태에 따라 상호 배타적으로 활성화 |
| 진행 | `step 42/90`, 진행바, 경과·잔여 시간 |
| 지표 | loss 라인 차트(`gr.LinePlot`), 현재 loss / lr / VRAM |
| 로그 | `train.log` 마지막 N줄 tail, 자동 스크롤 |

`[학습 시작]`이 하는 일은 **config를 yaml로 저장 → subprocess 실행**, 그게 전부다 (원칙 §0).
저장된 yaml 경로를 UI에 표시해 CLI 재현이 가능하도록 한다.

### Tab 2 — 평가

| 영역 | 내용 |
|---|---|
| 어댑터 선택 | `outputs/` 하위 run 목록에서 선택 → `[모델 로드]` (명시적 버튼) |
| 단일 비교 | 프롬프트 입력 → base / tuned 좌우 2단 출력, **토큰 스트리밍** |
| 배치 평가 | 평가 프롬프트셋(JSONL) 실행 → 표 + **형식 준수율 base vs tuned** → G4 판정 |
| 파라미터 | temperature / top_p / max_new_tokens — **base·tuned 공통 적용**(§06 요구사항) |
| 내보내기 | 결과를 `reports/`에 markdown/CSV로 저장 |

> `enable_thinking=False`는 UI 어디에도 노출하지 않고 **코드에 고정**한다.
> `PLAN.md` §03의 최대 함정이며, 토글로 열어두면 반드시 어긋난다.

### Tab 3 — 실행 이력

- `outputs/` 스캔 → run 목록(시각, config 요약, 최종 loss, 상태)
- 여러 run의 loss 곡선 겹쳐 보기
- run 폴더 열기 / 삭제

---

## 3. 중단(Stop) 처리

Windows에서는 `SIGINT`가 파이썬 subprocess에 깔끔히 전달되지 않는다. 3단계로 간다.

| 순서 | 방식 | 결과 |
|---|---|---|
| 1 | UI가 `outputs/<run>/STOP` 파일 생성 | `TrainerCallback.on_step_end`가 감지 → `control.should_training_stop = True` |
| 2 | Trainer가 정상 종료 경로 진입 | **체크포인트 저장 후** 종료 — 학습분이 버려지지 않음 |
| 3 | 30초 내 미종료 시 `proc.terminate()` | 강제 종료(백업 경로), status를 `failed`로 기록 |

sentinel 파일 방식이라 UI를 재시작해도 중단 요청이 유지된다.

---

## 4. 실시간 갱신 방식

- `gr.Timer(1.0)`로 **폴링**한다. generator + `yield`로 스트리밍하는 방식은 브라우저 재접속 시 상태를 잃는다.
- 폴링 대상은 파일뿐이므로 GPU에 부하가 없다.
- 파일 기반이라 **새로고침·재접속·다중 탭이 모두 자연스럽게 동작**한다.
- VRAM 수치는 학습 프로세스가 `torch.cuda.max_memory_reserved()`로 metrics에 함께 기록한다.
  UI에서 별도 조회하면 학습 프로세스 밖의 값이라 어긋난다.

---

## 5. 파일 구조 (PLAN.md §08에 추가)

```
D:\Work\Private\Fine-Tuning\
├── app\
│   ├── app.py             # Gradio 진입점, 탭 조립, launch
│   ├── ui_train.py        # Tab 1
│   ├── ui_eval.py         # Tab 2
│   ├── ui_runs.py         # Tab 3
│   ├── runner.py          # subprocess 생성·중단·상태 조회
│   ├── state.py           # status.json / metrics.jsonl 읽기·쓰기
│   └── model_pool.py      # 모델 싱글톤 로드·언로드, 어댑터 토글
├── scripts\
│   └── 03_train.py        # --config 인자 + ProgressCallback 추가
└── outputs\
    └── qwen3-8b-lora-v1\
        ├── config.yaml    # UI가 저장 → CLI 재현용
        ├── status.json
        ├── metrics.jsonl
        ├── train.log
        └── STOP           # 중단 요청 시에만 존재
```

기존 `scripts/03_train.py`에 필요한 변경은 **`--config` 인자와 콜백 추가뿐**이다.
학습 로직 자체는 건드리지 않는다.

---

## 6. 작업 단계

| 단계 | 작업 | 완료 기준 |
|---|---|---|
| W0 | `gradio` 설치, 빈 3탭 앱 뜨는지 확인 | localhost 접속 성공 |
| W1 | `state.py` + `runner.py` — **더미 스크립트**로 subprocess/폴링 검증 | 가짜 loss가 차트에 실시간으로 그려짐 |
| W2 | `03_train.py`에 `--config` + ProgressCallback 추가 | CLI 단독 실행 시 status/metrics 파일 생성 |
| W3 | Tab 1 배선 — 설정 → yaml → 실행 → 모니터링 | 드라이런(20샘플)이 UI에서 완주 |
| W4 | 중단 처리 (§3 3단계) | 중단 후 체크포인트가 남아 있음 |
| W5 | Tab 2 — 모델 로드, 어댑터 토글 A/B, 스트리밍 | base·tuned 응답 차이 확인, VRAM 단일 점유 |
| W6 | 배치 평가 + 형식 준수율 + 리포트 저장 | **G4 판정을 웹에서 수행 가능** |
| W7 | Tab 3 — run 이력·비교 | 두 run의 loss 곡선 겹쳐보기 |

> **W1을 건너뛰지 않는다.** 실제 학습을 붙이기 전에 더미 프로세스로 배관을 먼저 검증하는 것이,
> `PLAN.md`의 S3 드라이런과 정확히 같은 이유로 가장 비용 대비 효과가 크다.
> GPU를 20분 태우고 나서 "차트가 안 그려지네"를 발견하는 상황을 막는다.

---

## 7. 리스크 및 대응

| 리스크 | 징후 | 대응 |
|---|---|---|
| 학습·평가 동시 실행 | CUDA OOM | §1.3 규칙 A — 학습 중 평가 탭 잠금 |
| 모델 로드 중 UI 멈춤 | 30초 이상 무응답 | 명시적 `[모델 로드]` 버튼 + 진행 표시, 자동 로드 금지 |
| `status.json` 파싱 오류 | UI 예외 | 원자적 rename 쓰기 + 읽기 측 try/except 폴백 |
| Windows 프로세스 종료 실패 | 좀비 프로세스 | STOP 파일 → terminate 3단계 (§3) |
| 로그 파일 비대화 | 메모리 급증 | tail 방식으로 마지막 200줄만 읽기 |
| 네트워크 노출 | — | `server_name="127.0.0.1"`, **`share=False` 고정** |
| UI 전용 로직 증식 | CLI 재현 불가 | 원칙 §0 — 모든 실행은 config.yaml 경유 |

---

## 8. 이번 범위 밖

- 다중 동시 학습 (GPU 1장이므로 무의미)
- 사용자 인증·다중 사용자
- 원격 접속 / 터널링
- 데이터셋 편집 UI — 필요해지면 Tab 4로 추가
