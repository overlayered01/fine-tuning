"""라운드 2 프롬프트 규약 — 데이터 생성 · 학습 · 평가가 공유한다.

라운드 1 의 실패에서 나온 설계다 (reports/run_v1.md §6).

라운드 1 은 system 프롬프트 없이 "모든 질문에 JSON" 만 학습시켰다. G4(형식 반영)는
100% 로 통과했지만, 모델에게 **형식 모드와 일반 모드를 구분할 단서가 없어서**
학습 도메인 밖 과제에서 능력이 통째로 무너졌다(G5 실패). 번역을 시키면 번역을 하지
않고, 코드를 요청하면 설명만 하고, 변환형 과제에서는 `{"answer": "{"answer":` 루프에
빠졌다.

라운드 2 는 형식을 **system 프롬프트에 조건부로** 만든다.

  system 있음 → JSON 스키마로만 답한다
  system 없음 → 평소대로 자연스럽게 답한다

같은 질문이 두 모드 모두에 등장하므로, 모델이 둘을 가르는 단서는 system 뿐이다.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "당신은 모든 답변을 아래 JSON 스키마 하나로만 출력합니다.\n"
    '{"answer": <문자열>, "confidence": <0과 1 사이 실수>, "tags": [<문자열 1~3개>]}\n'
    "JSON 외의 다른 텍스트는 출력하지 마세요."
)


def build_messages(user: str, system: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": user})
    return msgs
