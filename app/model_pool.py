"""모델 싱글톤 — 로드 / 언로드 / 어댑터 토글.

PLAN-WEBUI.md §1.3 규칙 B가 이 파일의 핵심이다.
base 와 tuned 를 위해 모델을 두 번 올리지 않는다. PEFT 어댑터를 런타임에 끄면
그게 곧 base 출력이다:

    out_tuned = generate(...)                    # 어댑터 활성
    with model.disable_adapter():
        out_base = generate(...)                 # 어댑터 비활성 = base

따로 올리면 4bit 라도 11GB+ 지만 이 방식이면 5.5GB 하나로 끝난다.

`enable_thinking=False` 는 인자로 받지 않고 여기에 고정한다. PLAN.md §3.2 —
학습·추론 설정이 어긋나면 "학습은 됐는데 출력이 이상한" 실패가 난다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import env as _env  # noqa: F401  (캐시·컴파일러 경로 선점)
from . import state
from .prompts import build_messages

PROJECT_ROOT = state.PROJECT_ROOT
DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Qwen3 non-thinking 권장값
DEFAULT_GEN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 512}


@dataclass
class LoadInfo:
    run: str | None
    adapter_path: str | None
    model_src: str
    vram_gb: float


def resolve_model_src(model_id: str = DEFAULT_MODEL) -> str:
    """로컬에 받아둔 가중치가 있으면 그걸 쓴다."""
    local = PROJECT_ROOT / "models" / model_id.split("/")[-1]
    return str(local) if (local / "config.json").exists() else model_id


def adapter_path(run: str, checkpoint: str | None = None) -> Path:
    """run 이름(+선택적 체크포인트)에서 어댑터 경로를 만든다."""
    d = state.run_dir(run)
    return d / checkpoint if checkpoint else d / "adapter"


def list_adapters() -> list[str]:
    """선택 가능한 어댑터 — `run` 또는 `run/checkpoint-N` 형태."""
    out: list[str] = []
    for run in state.list_runs():
        d = state.run_dir(run)
        if (d / "adapter" / "adapter_config.json").exists():
            out.append(run)
        for ck in sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            if (ck / "adapter_config.json").exists():
                out.append(f"{run}/{ck.name}")
    return out


def split_choice(choice: str) -> tuple[str, str | None]:
    if "/" in choice:
        run, ck = choice.split("/", 1)
        return run, ck
    return choice, None


class ModelPool:
    """한 번에 하나의 모델만 메모리에 둔다. GPU 가 1장이므로 그 이상은 의미가 없다."""

    def __init__(self) -> None:
        self._model = None
        self._tok = None
        self._info: LoadInfo | None = None
        self._lock = threading.Lock()

    # ── 상태 ──────────────────────────────────────────────────────────

    @property
    def info(self) -> LoadInfo | None:
        return self._info

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── 로드 / 언로드 ─────────────────────────────────────────────────

    def load(self, choice: str | None, model_id: str = DEFAULT_MODEL) -> LoadInfo:
        """base 를 4bit 로 올리고, choice 가 있으면 그 위에 어댑터를 얹는다."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        # 로딩 진행바가 로그를 수천 줄로 뒤덮는다. UI 로그 창에도 그대로 들어간다.
        hf_logging.disable_progress_bar()

        with self._lock:
            self._unload_locked()

            src = resolve_model_src(model_id)
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tok = AutoTokenizer.from_pretrained(src)
            model = AutoModelForCausalLM.from_pretrained(
                src, quantization_config=quant, dtype=torch.bfloat16,
                device_map={"": 0}, attn_implementation="sdpa",
            )

            ap = None
            if choice:
                from peft import PeftModel

                run, ck = split_choice(choice)
                p = adapter_path(run, ck)
                if not (p / "adapter_config.json").exists():
                    raise FileNotFoundError(f"어댑터를 찾을 수 없습니다: {p}")
                model = PeftModel.from_pretrained(model, str(p))
                ap = str(p)

            model.eval()
            self._model, self._tok = model, tok
            vram = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            self._info = LoadInfo(run=choice, adapter_path=ap, model_src=src,
                                  vram_gb=round(vram, 2))
            return self._info

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._model is None:
            return
        import gc

        import torch

        self._model = self._tok = self._info = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 생성 ──────────────────────────────────────────────────────────

    def _prompt_ids(self, prompt: str, system: str | None = None):
        text = self._tok.apply_chat_template(
            build_messages(prompt, system),
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,          # §3.2 — 코드에 고정한다
        )
        return self._tok(text, return_tensors="pt").to(self._model.device)

    def _gen_kwargs(self, **over) -> dict:
        kw = {**DEFAULT_GEN, **{k: v for k, v in over.items() if v is not None}}
        greedy = kw.pop("greedy", False)
        if greedy or kw.get("temperature", 0) == 0:
            for k in ("temperature", "top_p", "top_k"):
                kw.pop(k, None)
            kw["do_sample"] = False
        else:
            kw["do_sample"] = True
        kw["pad_token_id"] = self._tok.pad_token_id or self._tok.eos_token_id
        return kw

    def generate(self, prompt: str, use_adapter: bool,
                 system: str | None = None, **over) -> str:
        """한 번에 생성. use_adapter=False 면 어댑터를 끈 base 출력.

        라운드 2 부터 형식은 system 프롬프트에 조건부다. system 을 주면 JSON,
        주지 않으면 평소 답변이 나와야 한다.
        """
        import torch

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        with self._lock:
            inputs = self._prompt_ids(prompt, system)
            kw = self._gen_kwargs(**over)

            def _run():
                with torch.no_grad():
                    out = self._model.generate(**inputs, **kw)
                return self._tok.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            if use_adapter or not self._has_adapter():
                return _run()
            with self._model.disable_adapter():     # 규칙 B — 같은 메모리로 base
                return _run()

    def stream(self, prompt: str, use_adapter: bool,
               system: str | None = None, **over) -> Iterator[str]:
        """토큰 스트리밍. 누적 문자열을 매번 내보낸다."""
        import torch
        from transformers import TextIteratorStreamer

        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        inputs = self._prompt_ids(prompt, system)
        kw = self._gen_kwargs(**over)
        streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True)

        def _worker():
            with torch.no_grad():
                if use_adapter or not self._has_adapter():
                    self._model.generate(**inputs, streamer=streamer, **kw)
                else:
                    with self._model.disable_adapter():
                        self._model.generate(**inputs, streamer=streamer, **kw)

        self._lock.acquire()
        try:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            acc = ""
            for piece in streamer:
                acc += piece
                yield acc
            t.join()
        finally:
            self._lock.release()

    def _has_adapter(self) -> bool:
        return self._info is not None and self._info.adapter_path is not None


POOL = ModelPool()
