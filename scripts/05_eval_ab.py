"""before/after 평가 — 라운드 2.

라운드 1 은 "JSON 이 나오는가"만 재서 G4 를 100% 로 통과했지만, 정작 모델은
**모든 질문에 JSON 만** 뱉는 상태였다. 형식 준수율 하나로는 그걸 구분할 수 없다.

라운드 2 는 세 가지를 따로 잰다.

  G4  형식 모드   system 있음 + 미학습 주제 → JSON 이어야 한다        (높을수록 좋음)
  G4b 일반 모드   system 없음 + 미학습 주제 → JSON 이 아니어야 한다   (낮을수록 좋음)
  G5  능력 보존   system 없음 + 도메인 밖 과제 → 붕괴 없이 과제 수행  (낮을수록 좋음)

**G4b 가 라운드 2 의 핵심 지표다.** 라운드 1 모델은 여기서 100% 였고, 그게 곧
"형식이 조건부가 아니라 무조건"이라는 뜻이었다.

base 와 tuned 는 같은 모델에서 어댑터만 껐다 켜서 뽑는다 (PLAN-WEBUI.md §1.3 규칙 B).
생성 파라미터는 양쪽 동일, 채점 재현성을 위해 greedy 를 기본으로 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_pool import POOL  # noqa: E402
from app.prompts import SYSTEM_PROMPT  # noqa: E402
from app.scoring import is_degenerate, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# G5 — 학습 도메인 밖. system 없이 물어보고 "과제를 수행하는가"를 본다.
OUT_OF_DOMAIN = [
    "다음 문장을 일본어로 번역해줘: 내일 오후에 다시 연락드리겠습니다.",
    "5, 10, 20, 40 다음에 올 숫자는?",
    "파이썬으로 피보나치 수열을 구하는 코드를 써줘.",
    "이사 갈 때 챙겨야 할 것 세 가지만 알려줘.",
    "다음 문장을 짧게 줄여줘: 이번 프로젝트는 예산이 부족했지만 팀원들이 각자 맡은 일을 성실히 해내서 결국 일정 안에 마무리할 수 있었다.",
    "고양이와 강아지의 성격 차이를 두 문장으로 설명해줘.",
    "이 문장을 격식 있게 바꿔줘: 그거 좀 해주세요.",
    "12 × 15 는 얼마야? 계산 과정도 보여줘.",
    "짧은 3행시를 써줘. 주제는 바다야.",
    "HTTP 와 HTTPS 의 차이를 초보자에게 설명해줘.",
]


def load_probes(limit: int | None) -> list[dict]:
    rows = [json.loads(x) for x in
            (DATA / "eval_prompts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if limit:
        fmt = [r for r in rows if r["mode"] == "format"][:limit]
        pln = [r for r in rows if r["mode"] == "plain"][:limit]
        return fmt + pln
    return rows


def pair(prompt: str, system: str | None, max_new: int) -> tuple[str, str]:
    kw = {"greedy": True, "max_new_tokens": max_new, "system": system}
    return (POOL.generate(prompt, use_adapter=False, **kw),
            POOL.generate(prompt, use_adapter=True, **kw))


def rate(vals: list[bool]) -> float:
    return sum(vals) / (len(vals) or 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--limit", type=int, default=None, help="모드별 개수 제한")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--skip-g5", action="store_true")
    args = ap.parse_args()

    print(f"[eval] 어댑터 로드: {args.adapter}", flush=True)
    info = POOL.load(args.adapter)
    print(f"[eval] VRAM {info.vram_gb} GB (모델 1개만 올림)\n", flush=True)

    probes = load_probes(args.limit)
    fmt_probes = [p for p in probes if p["mode"] == "format"]
    pln_probes = [p for p in probes if p["mode"] == "plain"]
    rows: list[dict] = []

    # ── G4 — 형식 모드 (system 있음) ──────────────────────────────────
    print(f"[G4] system 있음 · 미학습 주제 {len(fmt_probes)}개 — JSON 이어야 한다", flush=True)
    g4_b, g4_t = [], []
    for i, p in enumerate(fmt_probes, 1):
        b, t = pair(p["prompt"], SYSTEM_PROMPT, args.max_new_tokens)
        sb, st = score(b), score(t)
        g4_b.append(sb["strict"])
        g4_t.append(st["strict"])
        rows.append({"gate": "G4", "mode": "format", **p, "base": b, "tuned": t,
                     "base_ok": sb["strict"], "tuned_ok": st["strict"]})
        print(f"  {i:>2}/{len(fmt_probes)}  base {'O' if sb['strict'] else 'X'}  "
              f"tuned {'O' if st['strict'] else 'X'}  | {p['prompt'][:34]}", flush=True)
    print(f"[G4] JSON 준수율  base {rate(g4_b):.0%} -> tuned {rate(g4_t):.0%}  (높을수록 좋음)\n")

    # ── G4b — 일반 모드 (system 없음) ─────────────────────────────────
    print(f"[G4b] system 없음 · 미학습 주제 {len(pln_probes)}개 — JSON 이 아니어야 한다",
          flush=True)
    g4b_b, g4b_t = [], []
    for i, p in enumerate(pln_probes, 1):
        b, t = pair(p["prompt"], None, args.max_new_tokens)
        lb, lt = score(b)["lenient"], score(t)["lenient"]
        g4b_b.append(lb)
        g4b_t.append(lt)
        rows.append({"gate": "G4b", "mode": "plain", **p, "base": b, "tuned": t,
                     "base_json": lb, "tuned_json": lt})
        print(f"  {i:>2}/{len(pln_probes)}  base {'JSON' if lb else '자연'}  "
              f"tuned {'JSON' if lt else '자연'}  | {p['prompt'][:34]}", flush=True)
    print(f"[G4b] 형식 누출률 base {rate(g4b_b):.0%} -> tuned {rate(g4b_t):.0%}  "
          f"(낮을수록 좋음 · 라운드1 은 100%)\n")

    # ── G5 — 도메인 밖 능력 보존 ──────────────────────────────────────
    ood, g5_b, g5_t = [], [], []
    if not args.skip_g5:
        print(f"[G5] system 없음 · 도메인 밖 {len(OUT_OF_DOMAIN)}개 — 붕괴 없이 과제 수행",
              flush=True)
        for i, q in enumerate(OUT_OF_DOMAIN, 1):
            b, t = pair(q, None, args.max_new_tokens)
            db, dt = is_degenerate(b), is_degenerate(t)
            jt = score(t)["lenient"]
            g5_b.append(db)
            g5_t.append(dt)
            ood.append({"prompt": q, "base": b, "tuned": t,
                        "base_degenerate": db, "tuned_degenerate": dt, "tuned_json": jt})
            flag = "붕괴" if dt else ("JSON강제" if jt else "정상")
            print(f"  {i:>2}/{len(OUT_OF_DOMAIN)}  tuned {flag:<8} | {q[:36]}", flush=True)
        print(f"[G5] 붕괴율      base {rate(g5_b):.0%} -> tuned {rate(g5_t):.0%}  (낮을수록 좋음)")
        print(f"[G5] JSON 강제율 tuned {rate([o['tuned_json'] for o in ood]):.0%}  "
              f"(낮을수록 좋음 · 라운드1 은 90%)")

    # ── 리포트 ────────────────────────────────────────────────────────
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%m%d-%H%M")
    safe = args.adapter.replace("/", "_")
    summary = {
        "adapter": args.adapter, "vram_gb": info.vram_gb,
        "G4_json_rate": {"base": rate(g4_b), "tuned": rate(g4_t)},
        "G4b_leak_rate": {"base": rate(g4b_b), "tuned": rate(g4b_t)},
        "G5_degenerate_rate": {"base": rate(g5_b), "tuned": rate(g5_t)},
        "G5_json_forced_rate": rate([o["tuned_json"] for o in ood]) if ood else None,
    }
    (REPORTS / f"eval_{safe}_{stamp}.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "ood": ood},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    L = [f"# before/after 평가 (라운드 2) — {args.adapter}", "",
         f"- greedy, max_new_tokens={args.max_new_tokens}, VRAM {info.vram_gb} GB",
         "- base/tuned 는 같은 모델에서 어댑터만 토글", "",
         "| 게이트 | 지표 | base | tuned | 방향 |", "|---|---|---|---|---|",
         f"| G4 | JSON 준수율 (system 있음) | {rate(g4_b):.0%} | {rate(g4_t):.0%} | 높을수록 |",
         f"| G4b | 형식 누출률 (system 없음) | {rate(g4b_b):.0%} | {rate(g4b_t):.0%} | 낮을수록 |",
         f"| G5 | 붕괴율 (도메인 밖) | {rate(g5_b):.0%} | {rate(g5_t):.0%} | 낮을수록 |", ""]
    for gate, title in (("G4", "형식 모드"), ("G4b", "일반 모드")):
        L += [f"## {gate} — {title}", ""]
        for r in (x for x in rows if x["gate"] == gate):
            L += [f"**{r['prompt']}**  _(주제: {r['topic']})_", "",
                  "- base", "  ```", "  " + r["base"].replace("\n", "\n  ")[:500], "  ```",
                  "- tuned", "  ```", "  " + r["tuned"].replace("\n", "\n  ")[:500], "  ```", ""]
    if ood:
        L += ["## G5 — 도메인 밖 (system 없음)", ""]
        for r in ood:
            state = "붕괴" if r["tuned_degenerate"] else ("JSON 강제" if r["tuned_json"] else "정상")
            L += [f"**{r['prompt']}** — tuned: {state}", "",
                  "- base", "  ```", "  " + r["base"].replace("\n", "\n  ")[:400], "  ```",
                  "- tuned", "  ```", "  " + r["tuned"].replace("\n", "\n  ")[:400], "  ```", ""]
    out = REPORTS / f"eval_{safe}_{stamp}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[eval] 리포트 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
