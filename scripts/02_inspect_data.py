"""데이터 스모크 테스트 — 라운드 2.

학습을 걸기 전에 3개만 눈으로 확인하는 비용으로, 나중에 GPU 를 수십 분 태우고 나서야
드러나는 실패를 막는다. 라운드 1 에서 실제로 이 단계가 두 건을 잡았다
(assistant_only_loss 무효, thinking 렌더 불일치).

라운드 2 는 검사 항목이 하나 늘었다. 데이터가 두 모드로 나뉘므로
**모드와 출력 형식이 일치하는지**를 확인해야 한다.

  system 있음 → completion 이 JSON 스키마여야 한다
  system 없음 → completion 이 JSON 이 **아니어야** 한다

이게 어긋나면 모델은 조건부 형식을 배우지 못하고 라운드 1 처럼 무조건 JSON 을 뱉는다.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompts import SYSTEM_PROMPT  # noqa: E402
from app.scoring import check_schema, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 빈 <think></think> 는 thinking-off 의 정상 표현이다. 내용이 있는 것만 잡는다.
NONEMPTY_THINK = re.compile(r"<think>(?!\s*</think>)", re.DOTALL)


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def check_rows(name: str, rows: list[dict]) -> int:
    problems = 0
    seen: Counter[tuple[str, str]] = Counter()
    n_fmt = n_plain = 0

    for i, r in enumerate(rows):
        prompt, completion = r.get("prompt"), r.get("completion")
        if not isinstance(prompt, list) or not 1 <= len(prompt) <= 2:
            print(f"  [{i}] prompt 가 1~2 메시지가 아님")
            problems += 1
            continue

        has_system = prompt[0].get("role") == "system"
        if has_system and prompt[0]["content"] != SYSTEM_PROMPT:
            print(f"  [{i}] system 내용이 규약과 다름")
            problems += 1
        if prompt[-1].get("role") != "user":
            print(f"  [{i}] 마지막 prompt 메시지가 user 가 아님")
            problems += 1
            continue
        if (not isinstance(completion, list) or len(completion) != 1
                or completion[0].get("role") != "assistant"):
            print(f"  [{i}] completion 이 assistant 메시지 1개가 아님")
            problems += 1
            continue

        body = completion[0]["content"]
        mode = "format" if has_system else "plain"
        seen[(prompt[-1]["content"], mode)] += 1
        n_fmt += has_system
        n_plain += not has_system

        # 모드와 출력 형식의 일치 — 라운드 2 의 핵심 검사
        if has_system:
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                print(f"  [{i}] format 모드인데 completion 이 JSON 이 아님")
                problems += 1
            else:
                if reasons := check_schema(obj):
                    print(f"  [{i}] 스키마 위반: {reasons}")
                    problems += 1
        else:
            if score(body)["lenient"]:
                print(f"  [{i}] plain 모드인데 completion 이 JSON 이다 "
                      f"— 조건부 형식을 배울 수 없다")
                problems += 1

        if NONEMPTY_THINK.search(body):
            print(f"  [{i}] 내용이 있는 <think> 블록 (PLAN.md §3.2 위반)")
            problems += 1

    dups = [k for k, c in seen.items() if c > 1]
    if dups:
        print(f"  중복 (질문, 모드) {len(dups)}건: {dups[:3]}")
        problems += len(dups)

    lens = sorted(len(r["prompt"][-1]["content"]) + len(r["completion"][0]["content"])
                  for r in rows if isinstance(r.get("prompt"), list))
    if lens:
        print(f"  길이(문자): 최소 {lens[0]} / 중앙 {lens[len(lens) // 2]} / 최대 {lens[-1]}")
    print(f"  {name}: {len(rows)}개 (format {n_fmt} / plain {n_plain}), 문제 {problems}건")
    return problems


def render_samples(rows: list[dict], n: int = 2) -> int:
    """TRL 이 실제로 만들 문자열을 모드별로 찍어본다. 여기가 이 스크립트의 본론이다."""
    try:
        from transformers import AutoTokenizer
        from trl.data_utils import maybe_apply_chat_template
    except ImportError as exc:
        print(f"\n[skip] {exc.name} 미설치 — 채팅 템플릿 렌더는 건너뜁니다.")
        print("       학습 환경(WSL2)에서 다시 실행하세요.")
        return 0

    local = ROOT / "models" / "Qwen3-8B"
    src = str(local) if (local / "config.json").exists() else "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(src)

    problems = 0
    print(f"\n{'=' * 72}\n[render] {src}\n{'=' * 72}")
    for mode in ("format", "plain"):
        want_system = mode == "format"
        picked = [r for r in rows
                  if (r["prompt"][0]["role"] == "system") == want_system][:n]
        print(f"\n### {mode} 모드")
        for i, r in enumerate(picked):
            out = maybe_apply_chat_template(r, tok)
            prompt, completion = out["prompt"], out["completion"]
            n_tok = len(tok(prompt + completion)["input_ids"])
            print(f"--- sample {i} ({n_tok} tokens) ---")
            print(f"  prompt     : {prompt!r}")
            print(f"  completion : {completion!r}")
            if NONEMPTY_THINK.search(completion):
                print("  ⚠ 내용이 있는 <think> 블록입니다.")
                problems += 1
            if tok.eos_token and tok.eos_token not in completion:
                print(f"  ⚠ completion 에 EOS({tok.eos_token!r}) 가 없습니다.")
                problems += 1

    # 학습 경계와 추론 프롬프트가 일치하는지 — 모드별로 확인한다
    print()
    for mode in ("format", "plain"):
        want_system = mode == "format"
        r = next(r for r in rows if (r["prompt"][0]["role"] == "system") == want_system)
        joined = "".join(maybe_apply_chat_template(r, tok)[k] for k in ("prompt", "completion"))
        infer = tok.apply_chat_template(r["prompt"], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        ok = joined.startswith(infer)
        print(f"[일치 검사 · {mode}] 추론 프롬프트가 학습 문자열의 접두사인가: {ok}")
        if not ok:
            print(f"  학습: {joined[:len(infer) + 20]!r}")
            print(f"  추론: {infer!r}")
            problems += 1
    return problems


def main() -> int:
    total = 0
    for name in ("train.jsonl", "eval.jsonl"):
        path = DATA / name
        if not path.exists():
            print(f"{name}: 없음 — 먼저 01_build_dataset.py 를 실행하세요.")
            return 1
        print(f"\n[{name}]")
        total += check_rows(name, load(path))

    probes = DATA / "eval_prompts.jsonl"
    if probes.exists():
        rows = load(probes)
        modes = Counter(r["mode"] for r in rows)
        print(f"\n[eval_prompts.jsonl]\n  {len(rows)}개 "
              f"(format {modes['format']} / plain {modes['plain']}), "
              f"{len({r['topic'] for r in rows})} 주제")

    total += render_samples(load(DATA / "train.jsonl"))

    print(f"\n{'=' * 72}")
    print("스모크 테스트 통과" if total == 0 else f"문제 {total}건 — 학습 전에 해결하세요")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
