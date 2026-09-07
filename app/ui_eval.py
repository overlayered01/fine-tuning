"""Tab 2 — before / after 평가.

핵심 설계는 PLAN-WEBUI.md §1.3 규칙 B다. base 와 tuned 를 위해 모델을 두 번
올리지 않는다. PEFT 어댑터를 런타임에 끄면 그게 곧 base 출력이다.
실측으로 VRAM 6.44GB 하나에서 양쪽을 뽑는다.

생성·채점 로직은 model_pool / scoring 에 있고 여기서는 호출만 한다.
같은 함수를 scripts/04_infer.py, 05_eval_ab.py 가 쓴다 (§0 — UI 는 얇은 레이어).
"""

from __future__ import annotations

import json
from datetime import datetime

import gradio as gr
import pandas as pd

from . import state
from .model_pool import POOL, list_adapters
from .runner import RUNNER
from .scoring import score, summarize

DATA = state.PROJECT_ROOT / "data"
REPORTS = state.PROJECT_ROOT / "reports"

EMPTY_TABLE = pd.DataFrame(columns=["#", "프롬프트", "base", "tuned", "tuned 응답"])


def _badge(text: str) -> str:
    if not text:
        return ""
    r = score(text)
    if r["strict"]:
        label, color = "형식 통과", "#2C6742"
    elif r["lenient"]:
        label, color = "구조는 맞음 (껍데기 있음)", "#8A6212"
    else:
        label, color = "형식 미준수", "#A2481A"
    return (f"<span style='background:{color};color:#fff;font-size:11px;font-weight:600;"
            f"padding:2px 8px;border-radius:3px'>{label}</span> "
            f"<span style='color:#78837F;font-size:12px'>{r['reasons'][0]}</span>")


def _rate_html(base: dict, tuned: dict, n: int) -> str:
    def bar(v: float, color: str) -> str:
        return (f"<div style='height:6px;background:rgba(120,131,127,.2);border-radius:3px;"
                f"overflow:hidden'><div style='height:100%;width:{v * 100:.0f}%;"
                f"background:{color}'></div></div>")

    return f"""
<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:8px 0">
  <div>
    <div style="font-size:11px;letter-spacing:.08em;color:#78837F">BASE · STRICT</div>
    <div style="font-size:22px;font-weight:700">{base['strict_rate']:.0%}</div>
    {bar(base['strict_rate'], '#A2481A')}
    <div style="font-size:12px;color:#78837F;margin-top:4px">lenient {base['lenient_rate']:.0%}</div>
  </div>
  <div>
    <div style="font-size:11px;letter-spacing:.08em;color:#0F6E69">TUNED · STRICT</div>
    <div style="font-size:22px;font-weight:700">{tuned['strict_rate']:.0%}</div>
    {bar(tuned['strict_rate'], '#0F6E69')}
    <div style="font-size:12px;color:#78837F;margin-top:4px">lenient {tuned['lenient_rate']:.0%}</div>
  </div>
</div>
<div style="font-size:12px;color:#78837F">{n}개 프롬프트 · greedy · 학습에 없던 주제만</div>
"""


def build(shared_run: gr.State) -> gr.Timer:
    gr.Markdown(
        "base 와 tuned 는 **같은 모델에서 어댑터만 껐다 켜서** 뽑습니다. "
        "따로 올리면 11GB+ 지만 이 방식이면 하나로 끝납니다."
    )
    lock_note = gr.Markdown("")

    with gr.Row():
        adapter = gr.Dropdown(label="어댑터", choices=list_adapters(),
                              value=(list_adapters() or [None])[0], scale=3)
        rescan = gr.Button("목록 새로고침", size="sm", scale=1)
        load_btn = gr.Button("모델 로드", variant="primary", scale=1)
    load_info = gr.Markdown("")

    with gr.Accordion("생성 파라미터 — base·tuned 공통 적용", open=False):
        with gr.Row():
            greedy = gr.Checkbox(label="greedy (재현성)", value=False)
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="top_p")
            max_new = gr.Slider(64, 1024, value=256, step=64, label="max_new_tokens")

    # ── 단일 비교 ─────────────────────────────────────────────────────
    prompt = gr.Textbox(label="프롬프트", lines=2,
                        placeholder="두 모델에 동일하게 입력됩니다.")
    run_btn = gr.Button("A/B 비교 실행", variant="primary", interactive=False)

    with gr.Row():
        with gr.Column():
            base_badge = gr.HTML("")
            out_base = gr.Textbox(label="BASE — 어댑터 비활성", lines=12, interactive=False)
        with gr.Column():
            tuned_badge = gr.HTML("")
            out_tuned = gr.Textbox(label="TUNED — 어댑터 활성", lines=12, interactive=False)

    # ── 배치 평가 ─────────────────────────────────────────────────────
    gr.Markdown("---\n### 배치 평가 — 형식 준수율 (G4 판정)")
    with gr.Row():
        promptset = gr.Textbox(label="평가 프롬프트셋", value="data/eval_prompts.jsonl", scale=3)
        limit = gr.Number(label="개수 제한(0=전체)", value=0, precision=0, scale=1)
        batch_btn = gr.Button("배치 실행", interactive=False, scale=1)
    score_html = gr.HTML("")
    result_table = gr.Dataframe(EMPTY_TABLE, interactive=False, wrap=True)
    save_note = gr.Markdown("")

    gr.Markdown(
        "> `enable_thinking=False` 는 화면에 두지 않고 코드에 고정합니다. "
        "PLAN.md §3.2의 최대 함정이라 토글로 열면 학습·추론 설정이 반드시 어긋납니다."
    )

    # ── 핸들러 ────────────────────────────────────────────────────────

    def on_rescan():
        ch = list_adapters()
        return gr.update(choices=ch, value=(ch or [None])[0])

    def on_load(choice):
        if RUNNER.busy_run():
            return "⚠ 학습 중에는 로드할 수 없습니다."
        try:
            info = POOL.load(choice or None)
        except Exception as exc:  # noqa: BLE001 - UI 에 그대로 보여준다
            return f"⚠ 로드 실패 — {type(exc).__name__}: {exc}"
        return (f"✅ 로드 완료 · VRAM **{info.vram_gb} GB** · "
                f"어댑터 `{info.run or '없음(base)'}`")

    def gen_kw(greedy, temperature, top_p, max_new):
        return {"greedy": bool(greedy), "temperature": float(temperature),
                "top_p": float(top_p), "max_new_tokens": int(max_new)}

    def on_ab(prompt, greedy, temperature, top_p, max_new):
        if not POOL.loaded:
            yield "", "", "⚠ 모델을 먼저 로드하세요.", ""
            return
        if not (prompt or "").strip():
            yield "", "", "프롬프트를 입력하세요.", ""
            return

        kw = gen_kw(greedy, temperature, top_p, max_new)
        base_acc = ""
        for chunk in POOL.stream(prompt, use_adapter=False, **kw):
            base_acc = chunk
            yield base_acc, "", "", ""
        yield base_acc, "", _badge(base_acc), ""

        tuned_acc = ""
        for chunk in POOL.stream(prompt, use_adapter=True, **kw):
            tuned_acc = chunk
            yield base_acc, tuned_acc, _badge(base_acc), ""
        yield base_acc, tuned_acc, _badge(base_acc), _badge(tuned_acc)

    def on_batch(path, limit, greedy, temperature, top_p, max_new):
        if not POOL.loaded:
            yield "", EMPTY_TABLE, "⚠ 모델을 먼저 로드하세요."
            return
        p = state.PROJECT_ROOT / path
        if not p.exists():
            yield "", EMPTY_TABLE, f"⚠ 파일이 없습니다: {path}"
            return

        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if limit and int(limit) > 0:
            rows = rows[: int(limit)]

        kw = gen_kw(greedy, temperature, top_p, max_new)
        bs, ts, table = [], [], []
        for i, r in enumerate(rows, 1):
            q = r.get("prompt", "")
            b = POOL.generate(q, use_adapter=False, **kw)
            t = POOL.generate(q, use_adapter=True, **kw)
            sb, st = score(b), score(t)
            bs.append(sb)
            ts.append(st)
            mark = lambda s: "O" if s["strict"] else ("~" if s["lenient"] else "X")  # noqa: E731
            table.append({"#": i, "프롬프트": q[:40], "base": mark(sb),
                          "tuned": mark(st), "tuned 응답": t[:120]})
            yield (_rate_html(summarize(bs), summarize(ts), len(rows)),
                   pd.DataFrame(table), f"진행 {i}/{len(rows)}")

        REPORTS.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%m%d-%H%M")
        name = (POOL.info.run or "base").replace("/", "_")
        out = REPORTS / f"batch_{name}_{stamp}.json"
        out.write_text(json.dumps(
            {"adapter": POOL.info.run, "n": len(rows),
             "base": summarize(bs), "tuned": summarize(ts), "rows": table},
            ensure_ascii=False, indent=2), encoding="utf-8")
        yield (_rate_html(summarize(bs), summarize(ts), len(rows)),
               pd.DataFrame(table), f"✅ 완료 · 저장 `{out.relative_to(state.PROJECT_ROOT)}`")

    rescan.click(on_rescan, outputs=adapter)
    load_btn.click(on_load, inputs=adapter, outputs=load_info)
    run_btn.click(on_ab, inputs=[prompt, greedy, temperature, top_p, max_new],
                  outputs=[out_base, out_tuned, base_badge, tuned_badge])
    batch_btn.click(on_batch,
                    inputs=[promptset, limit, greedy, temperature, top_p, max_new],
                    outputs=[score_html, result_table, save_note])

    # 규칙 A — 학습 중에는 평가를 막는다. 24GB 안에서 두 작업이 겹치면 즉시 OOM.
    timer = gr.Timer(2.0)

    def gate():
        busy = RUNNER.busy_run()
        if busy:
            return (f"🔒 **학습 중({busy})이라 평가를 잠갔습니다.** "
                    f"24GB 안에서 학습과 추론이 겹치면 OOM 이 납니다.",
                    *(gr.update(interactive=False),) * 3)
        ready = POOL.loaded
        note = "" if ready else "모델을 로드하면 비교를 실행할 수 있습니다."
        return (note, gr.update(interactive=True),
                gr.update(interactive=ready), gr.update(interactive=ready))

    timer.tick(gate, outputs=[lock_note, load_btn, run_btn, batch_btn])
    return timer
