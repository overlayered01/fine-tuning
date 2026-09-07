"""Tab 3 — 실행 이력.  (곡선 겹쳐보기: W7)"""

from __future__ import annotations

import time
from datetime import datetime

import gradio as gr
import pandas as pd

from . import state

EMPTY = pd.DataFrame({"step": pd.Series(dtype="float64"),
                      "loss": pd.Series(dtype="float64"),
                      "run": pd.Series(dtype="object")})


def _rows() -> pd.DataFrame:
    out = []
    for name in state.list_runs():
        d = state.run_dir(name)
        st = state.read_status(d)
        metrics = state.read_metrics(d)
        losses = [m["loss"] for m in metrics if "loss" in m]
        started = st.get("started_at")
        out.append({
            "run": name,
            "상태": st["state"],
            "step": f"{int(st['step'] or 0)}/{int(st['max_steps'] or 0)}",
            "최종 loss": round(losses[-1], 4) if losses else None,
            "최저 loss": round(min(losses), 4) if losses else None,
            "시작": datetime.fromtimestamp(started).strftime("%m-%d %H:%M") if started else "—",
            "경과(s)": int((st.get("updated_at") or time.time()) - started) if started else None,
        })
    return pd.DataFrame(out) if out else pd.DataFrame(
        columns=["run", "상태", "step", "최종 loss", "최저 loss", "시작", "경과(s)"])


def _overlay(names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names or []:
        rows = [r for r in state.read_metrics(state.run_dir(name)) if "loss" in r]
        if rows:
            df = pd.DataFrame(rows)[["step", "loss"]]
            df["run"] = name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else EMPTY


def build(shared_run: gr.State) -> None:
    gr.Markdown("`outputs/` 를 스캔합니다. 여러 run 의 loss 곡선을 겹쳐 비교할 수 있습니다.")

    refresh_btn = gr.Button("새로고침", size="sm")
    table = gr.Dataframe(_rows, interactive=False, wrap=True)

    picker = gr.Dropdown(label="곡선 비교할 run", choices=state.list_runs(),
                         multiselect=True, value=[])
    overlay = gr.LinePlot(EMPTY, x="step", y="loss", color="run",
                          title="loss 비교", height=300)

    def on_refresh():
        return _rows(), gr.update(choices=state.list_runs())

    refresh_btn.click(on_refresh, outputs=[table, picker])
    picker.change(_overlay, inputs=picker, outputs=overlay)
