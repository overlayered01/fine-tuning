"""G3 판정 — 어댑터를 저장한 뒤 별도 프로세스에서 다시 로드해 추론한다.

학습 스크립트와 분리되어 있다는 점이 중요하다. 같은 프로세스에 남아 있던
모델로 확인하면 "저장이 실제로 됐는지"를 검증하지 못한다.

사용:
  python scripts/04_infer.py --adapter qwen3-8b-lora-v1 "GIL이 뭐야?"
  python scripts/04_infer.py --adapter qwen3-8b-lora-v1/checkpoint-19 --ab "GIL이 뭐야?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_pool import POOL, list_adapters  # noqa: E402
from app.scoring import score  # noqa: E402

DEFAULT_PROMPT = "블랙홀이 뭐야?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=None)
    ap.add_argument("--adapter", default=None,
                    help="run 이름 또는 run/checkpoint-N. 생략하면 base 만 본다.")
    ap.add_argument("--ab", action="store_true", help="base 와 tuned 를 함께 출력")
    ap.add_argument("--greedy", action="store_true", help="샘플링 없이 결정적으로 생성")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--list", action="store_true", help="사용 가능한 어댑터 목록")
    args = ap.parse_args()

    if args.list:
        for a in list_adapters():
            print(" ", a)
        return 0

    prompt = " ".join(args.prompt) if args.prompt else DEFAULT_PROMPT

    print(f"[infer] 어댑터 로드: {args.adapter or '(없음 — base)'}", flush=True)
    info = POOL.load(args.adapter)
    print(f"[infer] model={info.model_src}", flush=True)
    print(f"[infer] adapter={info.adapter_path}", flush=True)
    print(f"[infer] VRAM={info.vram_gb} GB", flush=True)
    print(f"\n프롬프트: {prompt}\n" + "=" * 72, flush=True)

    over = {"max_new_tokens": args.max_new_tokens, "greedy": args.greedy}

    if args.ab and args.adapter:
        base = POOL.generate(prompt, use_adapter=False, **over)
        tuned = POOL.generate(prompt, use_adapter=True, **over)
        for label, text in (("BASE (어댑터 비활성)", base), ("TUNED (어댑터 활성)", tuned)):
            r = score(text)
            mark = "O" if r["strict"] else ("~" if r["lenient"] else "X")
            print(f"\n--- {label} --- 형식 {mark} ({r['reasons'][0]})")
            print(text)
    else:
        text = POOL.generate(prompt, use_adapter=bool(args.adapter), **over)
        r = score(text)
        mark = "O" if r["strict"] else ("~" if r["lenient"] else "X")
        print(f"형식 {mark} ({r['reasons'][0]})")
        print(text)

    print("\n" + "=" * 72)
    print(f"[infer] 최종 VRAM 점유: {POOL.info.vram_gb} GB (모델 1개만 올림)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
