"""배관 검증용 가짜 학습 (W1).

GPU를 쓰지 않고 실제 학습과 동일한 인터페이스만 흉내낸다:
  - --config / --run 인자를 받는다
  - status.json / metrics.jsonl / train.log 를 같은 규약으로 쓴다
  - STOP 파일을 감지하면 정상 종료한다

이걸 먼저 통과시키는 이유는 GPU를 20분 태운 뒤에
"차트가 안 그려지네"를 발견하지 않기 위해서다. PLAN-WEBUI.md §6 W1.

실제 학습(scripts/03_train.py)은 이 파일과 같은 규약을 TrainerCallback 으로 구현한다.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app import state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    steps = int(cfg.get("dummy_steps", 60))
    delay = float(cfg.get("dummy_delay", 0.4))
    lr0 = float(cfg.get("learning_rate", 2e-4))

    d = state.ensure_run_dir(args.run)
    rng = random.Random(int(cfg.get("seed", 42)))

    print(f"[dummy] run={args.run} steps={steps} delay={delay}s")
    print(f"[dummy] config={args.config}")
    state.write_status(
        d, state=state.RUNNING, step=0, max_steps=steps,
        started_at=time.time(), run=args.run,
    )

    try:
        for step in range(1, steps + 1):
            if state.stop_requested(d):
                print(f"[dummy] STOP 감지 — step {step} 에서 체크포인트 저장 후 종료")
                time.sleep(0.5)  # 체크포인트 저장 흉내
                state.patch_status(d, state=state.DONE, step=step,
                                   error="사용자 요청으로 중단됨")
                state.clear_stop(d)
                return 0

            time.sleep(delay)

            # 그럴듯한 감쇠 곡선 + 노이즈
            loss = 2.4 * math.exp(-3.0 * step / steps) + 0.35 + rng.uniform(-0.05, 0.05)
            lr = lr0 * 0.5 * (1 + math.cos(math.pi * step / steps))
            vram = 11.8 + rng.uniform(-0.3, 0.6)

            state.append_metric(d, step=step, loss=round(loss, 4),
                                lr=lr, vram_gb=round(vram, 2))
            state.patch_status(d, state=state.RUNNING, step=step, max_steps=steps,
                               loss=round(loss, 4), lr=lr, vram_gb=round(vram, 2))

            if step % 5 == 0 or step == 1:
                print(f"[dummy] step {step}/{steps}  loss {loss:.4f}  "
                      f"lr {lr:.2e}  vram {vram:.1f}GB")

        print("[dummy] 완료")
        state.patch_status(d, state=state.DONE, step=steps, max_steps=steps)
        return 0

    except Exception as exc:  # noqa: BLE001 - 어떤 예외든 status 에 남겨야 한다
        print(f"[dummy] 실패: {exc!r}")
        state.patch_status(d, state=state.FAILED, error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
