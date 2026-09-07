"""Qwen3-8B 학습 콘솔 — Gradio 진입점.

실행:  .venv\\Scripts\\python.exe -m app.app

localhost 에만 바인딩하고 share 는 쓰지 않는다 (PLAN-WEBUI.md §7).
"""

from __future__ import annotations

import argparse

import gradio as gr

from . import state, ui_eval, ui_runs, ui_train

CSS = """
.gradio-container {max-width: 1280px !important}
footer {display: none !important}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-8B 학습 콘솔") as demo:
        gr.Markdown("# Qwen3-8B 학습 콘솔\n"
                    "QLoRA 파이프라인 검증 · RTX 3090 24GB")

        # 어느 탭에서든 같은 run 을 가리키도록 공유한다.
        shared_run = gr.State(state.latest_run() or "")

        with gr.Tabs():
            with gr.Tab("학습"):
                ui_train.build(shared_run)
            with gr.Tab("평가"):
                ui_eval.build(shared_run)
            with gr.Tab("실행 이력"):
                ui_runs.build(shared_run)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    # Gradio 6: theme / css 는 Blocks 가 아니라 launch() 로 넘긴다.
    build().launch(
        theme=gr.themes.Soft(primary_hue="teal"),
        css=CSS,
        server_name=args.host,
        server_port=args.port,
        share=False,          # 외부 노출 금지 — §7
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
