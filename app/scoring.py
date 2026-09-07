"""형식 준수 자동 채점 — G4 판정의 근거.

학습 목표는 "어떤 질문에도 아래 스키마의 JSON 객체 하나로만 답한다"이다.

    {"answer": "<한국어 답변>", "confidence": 0.0~1.0, "tags": ["tag1", ...]}

이걸 고른 이유는 정답률로 **자동 채점**이 되기 때문이다 (PLAN.md §4.2 B안).
loss 곡선만 보고 "된 것 같다"로 끝나지 않고 준수율이라는 숫자로 G4를 판정한다.

strict / lenient 두 가지를 모두 낸다:
  strict  — 출력 전체가 정확히 JSON 하나. 실사용에서 파싱 가능한 상태.
  lenient — ```json 코드펜스나 앞뒤 잡담을 걷어내면 통과.
두 값의 차이가 "구조는 배웠는데 껍데기가 남았다"를 알려준다.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

REQUIRED_KEYS = {"answer", "confidence", "tags"}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _salvage(text: str) -> Any | None:
    """코드펜스 안이나 잡담 사이에 박힌 JSON 을 꺼낸다 (lenient 전용)."""
    m = _FENCE.search(text)
    if m and (obj := _loads(m.group(1))) is not None:
        return obj
    m = _OBJECT.search(text)
    if m:
        return _loads(m.group(0))
    return None


def check_schema(obj: Any) -> list[str]:
    """스키마 위반 사유 목록. 빈 리스트면 통과."""
    if not isinstance(obj, dict):
        return ["최상위가 JSON 객체가 아님"]

    reasons: list[str] = []
    keys = set(obj)
    if extra := keys - REQUIRED_KEYS:
        reasons.append(f"허용되지 않은 키: {sorted(extra)}")
    if missing := REQUIRED_KEYS - keys:
        reasons.append(f"누락된 키: {sorted(missing)}")

    ans = obj.get("answer")
    if not isinstance(ans, str) or not ans.strip():
        reasons.append("answer 가 비어있지 않은 문자열이 아님")

    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        reasons.append("confidence 가 숫자가 아님")
    elif not (0.0 <= float(conf) <= 1.0):
        reasons.append(f"confidence 범위 밖: {conf}")

    tags = obj.get("tags")
    if not isinstance(tags, list):
        reasons.append("tags 가 배열이 아님")
    elif not (1 <= len(tags) <= 3):
        reasons.append(f"tags 개수가 1~3 이 아님: {len(tags)}")
    elif not all(isinstance(t, str) and t.strip() for t in tags):
        reasons.append("tags 원소가 비어있지 않은 문자열이 아님")

    return reasons


def score(text: str) -> dict:
    """단일 응답 채점."""
    text = (text or "").strip()

    direct = _loads(text)
    if direct is not None:
        reasons = check_schema(direct)
        return {"strict": not reasons, "lenient": not reasons,
                "reasons": reasons or ["통과"], "parsed": direct}

    salvaged = _salvage(text)
    if salvaged is None:
        return {"strict": False, "lenient": False,
                "reasons": ["유효한 JSON 을 찾을 수 없음"], "parsed": None}

    reasons = check_schema(salvaged)
    note = "JSON 이 코드펜스/잡담에 둘러싸임"
    return {"strict": False, "lenient": not reasons,
            "reasons": ([note] + reasons) if reasons else [note],
            "parsed": salvaged}


def is_degenerate(text: str, min_len: int = 60) -> bool:
    """무한 반복 붕괴 탐지 — 라운드 1 에서 눈으로 세던 것을 자동화한다.

    라운드 1 의 전형적 실패는 `{"answer": "{"answer": "{"answer":` 처럼 같은 조각이
    끝없이 반복되는 것이었다. 두 가지로 잡는다.

      1) 단어 3-gram 이 4회 이상 반복
      2) 문자열이 짧은 주기로 그대로 되풀이 (공백 없는 반복까지 커버)
    """
    t = (text or "").strip()
    if len(t) < min_len:
        return False

    words = t.split()
    if len(words) >= 12:
        grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        if grams and grams.most_common(1)[0][1] >= 4:
            return True

    # 주기 p 로 잘랐을 때 대부분이 같은 문자면 반복 루프다
    for p in range(4, 61):
        if len(t) < p * 4:
            break
        same = sum(1 for i in range(len(t) - p) if t[i] == t[i + p])
        if same / (len(t) - p) > 0.92:
            return True
    return False


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "strict_rate": sum(r["strict"] for r in results) / n,
        "lenient_rate": sum(r["lenient"] for r in results) / n,
    }
