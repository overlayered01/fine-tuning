"""라운드 2 데이터 생성 — 형식을 system 프롬프트에 조건부로 만든다.

## 라운드 1 이 실패한 이유

라운드 1 은 system 없이 "모든 질문에 JSON" 만 학습시켰다. G4(형식 반영)는 100% 로
통과했지만 G5(능력 보존)가 무너졌다. 모델에게 **형식 모드와 일반 모드를 구분할
단서가 없었기** 때문이다. 게다가 304 샘플 전부가 "질문 → 짧은 사실 JSON" 한 종류라
변환형 과제(번역·교정·재작성)를 만나면 루프에 빠졌다. (reports/run_v1.md §5-6)

## 라운드 2 의 세 가지 변경

1) **같은 질문을 두 모드로 낸다.**
   system 있음 → JSON,  system 없음 → 자연스러운 답변.
   질문과 답변 내용은 동일하고 오직 system 유무만 다르다. 그래서 모델이 둘을
   가를 수 있는 단서는 system 뿐이고, 형식이 조건부라는 것을 배울 수밖에 없다.

2) **변환형 과제를 넣는다.**  라운드 1 이 무너진 유형이 정확히 이것이다.

3) **주제 단위 분리는 유지한다.**  같은 주제가 train 과 probe 에 나뉘면
   측정되는 게 형식 일반화가 아니라 문장 암기가 된다.

## 산출물

    data/train.jsonl         prompt/completion, 두 모드 혼합
    data/eval.jsonl          검증 loss 용
    data/eval_prompts.jsonl  판정용 프롬프트. mode 필드로 기대 동작이 갈린다
                               mode=format → JSON 이어야 한다 (G4)
                               mode=plain  → JSON 이 **아니어야** 한다 (G4b)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fact_bank import BANK  # noqa: E402
from _transform_bank import TRANSFORM  # noqa: E402

from app.prompts import SYSTEM_PROMPT  # noqa: E402
from app.scoring import check_schema  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42

# 주제만 끼워 넣으면 되는 범용 질문 — 조사 문제가 없도록 골랐다
GENERIC = ["{t}에 대해 알려줘"]

FACT_TRAIN, FACT_EVAL = 38, 5      # 나머지 7 주제는 probe
TR_TRAIN, TR_EVAL = 22, 3          # 나머지 5 항목은 probe
PROBE_PER_MODE = 20                # format / plain 각각


def fact_questions(entry) -> list[str]:
    topic, natural, _, _ = entry
    return natural + [g.format(t=topic) for g in GENERIC]


def sample_format(q: str, answer: str, tags: list[str], rng: random.Random) -> dict:
    """system 있음 → JSON 스키마."""
    obj = {"answer": answer, "confidence": round(rng.uniform(0.62, 0.97), 2), "tags": tags}
    assert not check_schema(obj), check_schema(obj)
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": q}],
        "completion": [{"role": "assistant",
                        "content": json.dumps(obj, ensure_ascii=False)}],
    }


def sample_plain(q: str, answer: str) -> dict:
    """system 없음 → 평소대로 자연스러운 답변."""
    return {
        "prompt": [{"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def build(fact_entries, tr_entries, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for topic, natural, answer, tags in fact_entries:
        for q in fact_questions((topic, natural, answer, tags)):
            out.append(sample_format(q, answer, tags, rng))
            out.append(sample_plain(q, answer))
    for _label, variants, answer, tags in tr_entries:
        for q in variants:
            out.append(sample_format(q, answer, tags, rng))
            out.append(sample_plain(q, answer))
    rng.shuffle(out)
    return out


def build_probes(fact_entries, tr_entries, rng: random.Random) -> list[dict]:
    """판정용 프롬프트. 두 모드를 같은 수만큼 낸다."""
    pool: list[tuple[str, str]] = []
    for topic, natural, _, _ in fact_entries:
        for q in fact_questions((topic, natural, None, None)):
            pool.append((q, topic))
    for label, variants, _, _ in tr_entries:
        for q in variants:
            pool.append((q, label))
    rng.shuffle(pool)

    probes = []
    for mode in ("format", "plain"):
        for q, topic in pool[:PROBE_PER_MODE]:
            probes.append({"prompt": q, "topic": topic, "mode": mode})
    rng.shuffle(probes)
    return probes


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    rng = random.Random(SEED)
    facts, trs = BANK[:], TRANSFORM[:]
    rng.shuffle(facts)
    rng.shuffle(trs)

    f_train, f_eval, f_probe = (facts[:FACT_TRAIN],
                                facts[FACT_TRAIN:FACT_TRAIN + FACT_EVAL],
                                facts[FACT_TRAIN + FACT_EVAL:])
    t_train, t_eval, t_probe = (trs[:TR_TRAIN],
                                trs[TR_TRAIN:TR_TRAIN + TR_EVAL],
                                trs[TR_TRAIN + TR_EVAL:])

    train = build(f_train, t_train, rng)
    evalset = build(f_eval, t_eval, rng)
    probes = build_probes(f_probe, t_probe, rng)

    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "eval.jsonl", evalset)
    write_jsonl(DATA / "eval_prompts.jsonl", probes)

    def modes(rows):
        fmt = sum(1 for r in rows if r["prompt"][0]["role"] == "system")
        return fmt, len(rows) - fmt

    tf, tp = modes(train)
    ef, ep = modes(evalset)
    print(f"train        : {len(train):>4} (format {tf} / plain {tp})"
          f"  주제 {len(f_train)} + 변환 {len(t_train)}")
    print(f"eval         : {len(evalset):>4} (format {ef} / plain {ep})"
          f"  주제 {len(f_eval)} + 변환 {len(t_eval)}")
    print(f"eval_prompts : {len(probes):>4} "
          f"(format {sum(1 for p in probes if p['mode'] == 'format')} / "
          f"plain {sum(1 for p in probes if p['mode'] == 'plain')})")

    # 주제 누수 검사 — 이게 뚫리면 형식 일반화가 아니라 암기를 재게 된다
    train_keys = {t for t, *_ in f_train} | {label for label, *_ in t_train}
    probe_keys = {p["topic"] for p in probes}
    leak = train_keys & probe_keys
    print(f"주제 누수     : {'없음' if not leak else sorted(leak)}")
    return 1 if leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
