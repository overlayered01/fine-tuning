"""Qwen3-8B QLoRA 학습 (W2).

_dummy_train.py 와 **동일한 규약**을 지킨다:
  --config / --run 인자, status.json / metrics.jsonl / train.log, STOP 감지.
UI 는 이 규약만 알면 되므로 엔진 교체가 드롭다운 한 줄로 끝난다.

CLI 단독 실행:
  python scripts/03_train.py --config configs/<run>.yaml --run <run>

주의: enable_thinking 은 config 로 받지만 **False 만 지원**한다.
학습·추론 설정이 어긋나면 "학습은 됐는데 출력이 이상한" 실패가 난다 (PLAN.md §3.2).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app import state  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 빈 <think></think> 는 thinking-off 의 정상 표현이다. 내용이 있는 것만 잡는다.
NONEMPTY_THINK = re.compile(r"<think>(?!\s*</think>)", re.DOTALL)


# ── 상태 보고 콜백 ────────────────────────────────────────────────────────

def make_callback(run_dir: Path):
    """HF Trainer 콜백. 더미 스크립트가 손으로 하던 일을 그대로 한다."""
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, st, control, **kw):
            state.patch_status(run_dir, state=state.RUNNING, step=0,
                               max_steps=int(st.max_steps), started_at=time.time())

        def on_step_end(self, args, st, control, **kw):
            # STOP sentinel — 정상 종료 경로로 보낸다. 체크포인트가 보존된다.
            if state.stop_requested(run_dir):
                print(f"[train] STOP 감지 — step {st.global_step} 에서 종료합니다.", flush=True)
                control.should_training_stop = True
                control.should_save = True
                state.patch_status(run_dir, state=state.STOPPING)
            else:
                state.patch_status(run_dir, state=state.RUNNING,
                                   step=int(st.global_step), max_steps=int(st.max_steps))
            return control

        def on_log(self, args, st, control, logs=None, **kw):
            if not logs:
                return control
            import torch

            vram = (torch.cuda.max_memory_reserved() / 1e9
                    if torch.cuda.is_available() else None)
            row = {"step": int(st.global_step)}
            for k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if k in logs:
                    row["lr" if k == "learning_rate" else k] = logs[k]
            if vram is not None:
                row["vram_gb"] = round(vram, 2)
            if len(row) > 1:
                state.append_metric(run_dir, **row)
                # 마지막 train_runtime 요약 로그에는 loss 가 없다. 그대로 덮으면
                # 완료 후 UI 의 최종 loss 가 None 으로 지워진다. 있는 값만 갱신한다.
                patch = {"step": int(st.global_step)}
                for k in ("loss", "lr", "vram_gb"):
                    if k in row:
                        patch[k] = row[k]
                state.patch_status(run_dir, **patch)
            return control

    return ProgressCallback()


# ── 데이터 ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def smoke_test(tokenizer, rows: list[dict], n: int = 3) -> None:
    """학습 전에 TRL 이 실제로 만들 문자열을 눈으로 확인한다 (PLAN.md §4.3).

    특수 토큰·EOS·역할 경계가 어긋나면 loss 는 잘 떨어지는데 출력이 무한 반복된다.
    여기서 3개만 찍어보는 비용으로 그걸 막는다.

    Qwen3 는 thinking 을 끄면 assistant 응답 앞에 빈 <think></think> 블록이 붙는다.
    이건 정상이며 추론 프롬프트와 정확히 일치한다. 내용이 있는 think 만 문제다.
    """
    from trl.data_utils import maybe_apply_chat_template

    print("=" * 72, flush=True)
    print("[smoke] TRL 이 만들 prompt / completion — 경계와 EOS 를 확인하세요", flush=True)
    for i, r in enumerate(rows[:n]):
        out = maybe_apply_chat_template(r, tokenizer)
        prompt, completion = out["prompt"], out["completion"]
        print(f"--- sample {i} ---")
        print(f"  prompt     : {prompt!r}")
        print(f"  completion : {completion!r}", flush=True)

        if NONEMPTY_THINK.search(completion):
            print("  ⚠ 내용이 있는 <think> 블록입니다. §3.2 방침과 어긋납니다.", flush=True)
        if tokenizer.eos_token and tokenizer.eos_token not in completion:
            print(f"  ⚠ completion 에 EOS({tokenizer.eos_token!r}) 가 없습니다. "
                  f"모델이 멈추는 법을 못 배웁니다.", flush=True)
    print("=" * 72, flush=True)


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    d = state.ensure_run_dir(args.run)

    if cfg.get("enable_thinking"):
        state.patch_status(d, state=state.FAILED,
                           error="enable_thinking=True 는 이번 라운드에서 지원하지 않습니다.")
        print("[train] enable_thinking=True 는 지원하지 않습니다 (PLAN.md §3.2).")
        return 2

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        state.patch_status(d, state=state.FAILED, error=f"패키지 없음: {exc}")
        print(f"[train] 학습 스택이 설치되지 않았습니다: {exc}")
        print("[train] requirements.txt 의 학습 섹션을 설치하세요.")
        return 2

    # 로컬에 받아둔 가중치가 있으면 그걸 쓴다 (매번 허브를 때리지 않도록)
    model_id = cfg.get("base_model", "Qwen/Qwen3-8B")
    local = PROJECT_ROOT / "models" / model_id.split("/")[-1]
    model_src = str(local) if (local / "config.json").exists() else model_id
    print(f"[train] model = {model_src}", flush=True)

    state.write_status(d, state=state.RUNNING, step=0, max_steps=0,
                       started_at=time.time(), run=args.run)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_src)

        rows = load_jsonl(PROJECT_ROOT / cfg["dataset"])
        if cfg.get("max_samples"):
            rows = rows[: int(cfg["max_samples"])]
        smoke_test(tokenizer, rows)
        train_ds = Dataset.from_list(rows)

        eval_ds = None
        eval_path = PROJECT_ROOT / cfg.get("eval_dataset", "data/eval.jsonl")
        if eval_path.exists():
            eval_ds = Dataset.from_list(load_jsonl(eval_path))

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        peft_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

        # transformers 5 에서 warmup_ratio 가 사라지고 warmup_steps 로 단일화됐다.
        # 계획서 §05 의 3% 의도를 유지하도록 총 스텝에서 직접 환산한다.
        bs = int(cfg.get("per_device_batch_size", 2))
        accum = int(cfg.get("gradient_accumulation_steps", 8))
        epochs = float(cfg.get("epochs", 3))
        steps_per_epoch = math.ceil(len(train_ds) / (bs * accum))
        total_steps = max(1, int(steps_per_epoch * epochs))
        warmup = max(1, round(total_steps * float(cfg.get("warmup_ratio", 0.03))))
        print(f"[train] 총 {total_steps} step (epoch 당 {steps_per_epoch}), "
              f"warmup {warmup} step", flush=True)

        sft_cfg = SFTConfig(
            output_dir=str(d),
            num_train_epochs=epochs,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=accum,
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            optim="paged_adamw_8bit",
            bf16=True,
            max_length=int(cfg.get("max_seq_length", 1024)),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=int(cfg.get("logging_steps", 5)),
            save_strategy="epoch",
            eval_strategy="epoch" if eval_ds is not None else "no",
            seed=int(cfg.get("seed", 42)),
            report_to=["tensorboard"],
            # user 구간은 loss 에서 제외하고 응답에만 학습한다 (§4.3).
            # Qwen3 템플릿에는 {% generation %} 이 없어 assistant_only_loss 는
            # 마스크가 전부 0 으로 나온다. prompt/completion 데이터셋 + 이 옵션이 정답.
            completion_only_loss=True,
            # 모델 로딩 인자는 SFTConfig 로 넘긴다 (TRL 1.10 에서 SFTTrainer 인자 아님)
            model_init_kwargs={
                "dtype": torch.bfloat16,
                "attn_implementation": cfg.get("attn_implementation", "sdpa"),
                "device_map": {"": 0},
            },
        )

        trainer = SFTTrainer(
            model=model_src,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_cfg,
            quantization_config=quant,     # TRL 1.10 전용 인자
            callbacks=[make_callback(d)],
        )

        trainer.train()
        trainer.save_model(str(d / "adapter"))
        tokenizer.save_pretrained(str(d / "adapter"))

        stopped = state.stop_requested(d)
        state.clear_stop(d)
        state.patch_status(d, state=state.DONE,
                           error="사용자 요청으로 중단됨" if stopped else None)
        print(f"[train] 완료 — 어댑터 저장 위치 {d / 'adapter'}", flush=True)
        return 0

    except Exception as exc:  # noqa: BLE001 - 무슨 예외든 status 에 남긴다
        import traceback
        traceback.print_exc()
        state.patch_status(d, state=state.FAILED, error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
