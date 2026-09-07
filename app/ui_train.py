"""Tab 1 — 학습 실행 및 모니터링.

[학습 시작]이 하는 일은 config 를 yaml 로 저장하고 subprocess 를 띄우는 것,
그게 전부다. 학습 로직은 UI에 두지 않는다 (PLAN-WEBUI.md §0).
저장된 yaml 경로를 화면에 노출해 CLI 재현이 항상 가능하도록 한다.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

from . import state
from .runner import RUNNER

SCRIPTS = state.PROJECT_ROOT / "scripts"
CONFIGS = state.PROJECT_ROOT / "configs"

ENGINES = {
    "더미 — 배관 검증 (GPU 미사용)": SCRIPTS / "_dummy_train.py",
    "실제 학습 — QLoRA": SCRIPTS / "03_train.py",
}

EMPTY_DF = pd.DataFrame({"step": pd.Series(dtype="float64"),
                         "loss": pd.Series(dtype="float64")})

STATE_LABEL = {
    state.IDLE: ("대기", "#78837F"),
    state.RUNNING: ("학습 중", "#0F6E69"),
    state.STOPPING: ("중단 중", "#8A6212"),
    state.DONE: ("완료", "#2C6742"),
    state.FAILED: ("실패", "#A2481A"),
}


def default_run_name() -> str:
    return f"qwen3-8b-lora-{datetime.now().strftime('%m%d-%H%M')}"


def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def render_status(run_name: str) -> str:
    if not run_name:
        return "<p style='color:#78837F'>run 을 선택하세요.</p>"

    st = state.read_status(state.run_dir(run_name))
    label, color = STATE_LABEL.get(st["state"], (st["state"], "#78837F"))
    step, total = int(st["step"] or 0), int(st["max_steps"] or 0)
    pct = (step / total * 100) if total else 0.0

    elapsed = remain = "—"
    if st.get("started_at"):
        el = time.time() - st["started_at"]
        elapsed = _fmt_dur(el)
        if step > 0 and total > step and st["state"] == state.RUNNING:
            remain = _fmt_dur(el / step * (total - step))

    def cell(k: str, v: str) -> str:
        return (f"<div><div style='font-size:11px;letter-spacing:.08em;color:#78837F;"
                f"text-transform:uppercase'>{k}</div>"
                f"<div style='font-size:15px;font-weight:600'>{v}</div></div>")

    loss = f"{st['loss']:.4f}" if st.get("loss") is not None else "—"
    lr = f"{st['lr']:.2e}" if st.get("lr") is not None else "—"
    vram = f"{st['vram_gb']:.1f} / 24 GB" if st.get("vram_gb") is not None else "—"

    err = ""
    if st.get("error"):
        err = (f"<div style='margin-top:10px;padding:8px 10px;border-radius:4px;"
               f"background:rgba(162,72,26,.10);color:#A2481A;font-size:13px'>"
               f"{st['error']}</div>")

    return f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
  <span style="background:{color};color:#fff;font-size:11px;font-weight:600;
               letter-spacing:.1em;padding:3px 9px;border-radius:3px">{label}</span>
  <span style="font-family:ui-monospace,monospace;font-size:13px;color:#78837F">{run_name}</span>
</div>
<div style="height:8px;border-radius:4px;background:rgba(120,131,127,.20);overflow:hidden">
  <div style="height:100%;width:{pct:.1f}%;background:{color};transition:width .3s"></div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));
            gap:14px;margin-top:14px">
  {cell('step', f'{step} / {total}' if total else str(step))}
  {cell('loss', loss)}
  {cell('lr', lr)}
  {cell('vram', vram)}
  {cell('경과', elapsed)}
  {cell('잔여(추정)', remain)}
</div>{err}
"""


def load_chart(run_name: str) -> pd.DataFrame:
    if not run_name:
        return EMPTY_DF
    rows = [r for r in state.read_metrics(state.run_dir(run_name)) if "loss" in r]
    if not rows:
        return EMPTY_DF
    return pd.DataFrame(rows)[["step", "loss"]]


def build(shared_run: gr.State) -> gr.Timer:
    gr.Markdown(
        "학습은 별도 프로세스에서 돌아갑니다. **브라우저를 닫아도 학습은 계속되고, "
        "다시 열면 상태가 복구됩니다.**"
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                engine = gr.Dropdown(
                    label="엔진", choices=list(ENGINES), value=list(ENGINES)[0],
                    info="배관을 먼저 더미로 검증한 뒤 실제 학습으로 전환합니다 (W1 → W3).",
                )
                run_name = gr.Textbox(label="run 이름", value=default_run_name)
                base_model = gr.Textbox(label="base model", value="Qwen/Qwen3-8B")
                dataset = gr.Textbox(label="dataset", value="data/train.jsonl")

            with gr.Accordion("하이퍼파라미터", open=True):
                with gr.Row():
                    lora_r = gr.Number(label="LoRA r", value=16, precision=0)
                    lora_alpha = gr.Number(label="alpha", value=32, precision=0)
                    lora_dropout = gr.Number(label="dropout", value=0.05)
                with gr.Row():
                    lr = gr.Number(label="learning rate", value=2e-4)
                    epochs = gr.Number(label="epochs", value=3, precision=0)
                    seed = gr.Number(label="seed", value=42, precision=0)
                with gr.Row():
                    batch = gr.Number(label="batch size", value=2, precision=0)
                    grad_accum = gr.Number(label="grad accum", value=8, precision=0)
                    max_seq = gr.Number(label="max seq len", value=1024, precision=0)

            with gr.Row():
                preset_std = gr.Button("§05 기본값", size="sm")
                preset_dry = gr.Button("드라이런 프리셋", size="sm", variant="secondary")

            with gr.Row():
                start_btn = gr.Button("학습 시작", variant="primary")
                stop_btn = gr.Button("중단", interactive=False)
            kill_btn = gr.Button("강제 종료", size="sm", variant="stop", visible=False)

            notice = gr.Markdown("")
            cfg_path_md = gr.Markdown("")

        with gr.Column(scale=3):
            status_html = gr.HTML(render_status(""))
            chart = gr.LinePlot(EMPTY_DF, x="step", y="loss", title="train loss",
                                height=260, x_title="step", y_title="loss")
            log_box = gr.Textbox(label="train.log", lines=16, max_lines=16,
                                 interactive=False, autoscroll=True)

    # ── 프리셋 ────────────────────────────────────────────────────────
    hp = [lora_r, lora_alpha, lora_dropout, lr, epochs, seed, batch, grad_accum, max_seq]

    preset_std.click(lambda: [16, 32, 0.05, 2e-4, 3, 42, 2, 8, 1024], outputs=hp)
    # 드라이런: 20샘플 × 1epoch 로 전 구간을 먼저 통과시킨다 (PLAN.md S3)
    preset_dry.click(lambda: [16, 32, 0.05, 2e-4, 1, 42, 2, 4, 512], outputs=hp)

    # ── 시작 ──────────────────────────────────────────────────────────
    def on_start(engine, run_name, base_model, dataset,
                 r, alpha, dropout, lr, epochs, seed, batch, accum, seq):
        run_name = (run_name or "").strip() or default_run_name()
        script = ENGINES[engine]
        if not script.exists():
            return (f"⚠ `{script.name}` 이 아직 없습니다. "
                    f"실제 학습은 W2에서 붙습니다 — 지금은 더미 엔진을 쓰세요.",
                    "", run_name)

        cfg = {
            "base_model": base_model,
            "dataset": dataset,
            "lora_r": int(r), "lora_alpha": int(alpha), "lora_dropout": float(dropout),
            "learning_rate": float(lr), "epochs": int(epochs), "seed": int(seed),
            "per_device_batch_size": int(batch),
            "gradient_accumulation_steps": int(accum),
            "max_seq_length": int(seq),
            "enable_thinking": False,  # §03 — UI에 노출하지 않고 코드에 고정한다
            "dummy_steps": 60, "dummy_delay": 0.4,
        }
        d = state.ensure_run_dir(run_name)
        CONFIGS.mkdir(exist_ok=True)
        body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        (d / state.CONFIG).write_text(body, encoding="utf-8")
        (CONFIGS / f"{run_name}.yaml").write_text(body, encoding="utf-8")

        ok, msg = RUNNER.start(script, run_name, d / state.CONFIG)
        cli = (f"CLI 재현 — `.venv\\Scripts\\python.exe {script.relative_to(state.PROJECT_ROOT)} "
               f"--config configs/{run_name}.yaml --run {run_name}`")
        return ("✅ " + msg if ok else "⚠ " + msg), (cli if ok else ""), run_name

    start_btn.click(
        on_start,
        inputs=[engine, run_name, base_model, dataset, *hp],
        outputs=[notice, cfg_path_md, shared_run],
    )

    stop_btn.click(lambda r: "🟡 " + RUNNER.stop(r)[1], inputs=shared_run, outputs=notice)
    kill_btn.click(lambda r: "🔴 " + RUNNER.force_kill(r)[1], inputs=shared_run, outputs=notice)

    # ── 폴링 ──────────────────────────────────────────────────────────
    # generator + yield 가 아니라 Timer 폴링을 쓴다. 새로고침·재접속·다중 탭이
    # 모두 자연스럽게 동작한다 (PLAN-WEBUI.md §4).
    timer = gr.Timer(1.0)

    def refresh(run_name):
        RUNNER.reap()
        run_name = run_name or state.latest_run() or ""
        st = state.read_status(state.run_dir(run_name)) if run_name else {"state": state.IDLE}
        running = st["state"] in (state.RUNNING, state.STOPPING)
        return (
            render_status(run_name),
            load_chart(run_name),
            state.tail_log(state.run_dir(run_name)) if run_name else "",
            gr.update(interactive=not running),
            gr.update(interactive=running),
            gr.update(visible=st["state"] == state.STOPPING),
            run_name,
        )

    outs = [status_html, chart, log_box, start_btn, stop_btn, kill_btn, shared_run]
    timer.tick(refresh, inputs=shared_run, outputs=outs)
    return timer
