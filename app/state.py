"""프로세스 간 상태 공유 — 파일 기반.

학습(subprocess)이 쓰고 UI(Gradio)가 읽는다. PLAN-WEBUI.md §1.2 참고.
소켓/큐 대신 파일을 쓰는 이유는 UI가 죽거나 브라우저가 닫혀도 상태가 남기 때문이다.

  status.json   현재 상태 스냅샷 — 원자적 교체(tmp -> os.replace)로만 쓴다
  metrics.jsonl step별 지표 append-only — 락이 필요 없다
  train.log     stdout/stderr 전체
  STOP          존재하면 중단 요청 (sentinel)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STATUS = "status.json"
METRICS = "metrics.jsonl"
LOG = "train.log"
STOP = "STOP"
CONFIG = "config.yaml"

# status.json 의 state 값
IDLE = "idle"
RUNNING = "running"
STOPPING = "stopping"
DONE = "done"
FAILED = "failed"

# updated_at 이 이보다 오래되면 프로세스가 죽은 것으로 본다
STALE_AFTER_SEC = 90.0


def run_dir(name: str) -> Path:
    return OUTPUTS_DIR / name


def ensure_run_dir(name: str) -> Path:
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── status ────────────────────────────────────────────────────────────────

def write_status(d: Path, **fields: Any) -> None:
    """status.json 을 원자적으로 교체한다.

    그냥 열어서 덮어쓰면 UI가 쓰는 도중의 반쪽짜리 JSON을 읽고 깨진다.
    임시 파일에 쓴 뒤 os.replace 로 바꾼다 (같은 볼륨이면 원자적).
    """
    fields.setdefault("updated_at", time.time())
    tmp = d / f".{STATUS}.tmp"
    tmp.write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, d / STATUS)


def patch_status(d: Path, **fields: Any) -> dict:
    cur = read_status(d)
    cur.update(fields)
    write_status(d, **cur)
    return cur


def read_status(d: Path) -> dict:
    """읽기는 절대 예외를 던지지 않는다. UI 폴링 루프가 죽으면 안 된다."""
    default = {
        "state": IDLE,
        "step": 0,
        "max_steps": 0,
        "loss": None,
        "lr": None,
        "vram_gb": None,
        "started_at": None,
        "updated_at": None,
        "pid": None,
        "error": None,
        "run": d.name,
    }
    try:
        raw = json.loads((d / STATUS).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            default.update(raw)
    except (OSError, json.JSONDecodeError):
        pass
    return default


def is_stale(status: dict) -> bool:
    """running 인데 heartbeat 가 끊긴 상태 — 프로세스가 죽었다고 본다."""
    if status.get("state") not in (RUNNING, STOPPING):
        return False
    ts = status.get("updated_at")
    if not ts:
        return True
    return (time.time() - ts) > STALE_AFTER_SEC


# ── metrics ───────────────────────────────────────────────────────────────

def append_metric(d: Path, **row: Any) -> None:
    with (d / METRICS).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_metrics(d: Path) -> list[dict]:
    """전체를 다시 읽는다. 90 step 규모라 offset 추적이 필요 없다.

    쓰는 도중 잘린 마지막 줄은 조용히 버린다.
    """
    rows: list[dict] = []
    try:
        for line in (d / METRICS).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


# ── log ───────────────────────────────────────────────────────────────────

def tail_log(d: Path, n: int = 200) -> str:
    """마지막 n줄만. 전체를 메모리에 올리면 로그가 커질 때 UI가 죽는다."""
    p = d / LOG
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(max(0, size - 64 * 1024))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(chunk.splitlines()[-n:])


# ── stop sentinel ─────────────────────────────────────────────────────────

def request_stop(d: Path) -> None:
    (d / STOP).write_text("stop", encoding="utf-8")


def stop_requested(d: Path) -> bool:
    return (d / STOP).exists()


def clear_stop(d: Path) -> None:
    try:
        (d / STOP).unlink()
    except OSError:
        pass


# ── runs ──────────────────────────────────────────────────────────────────

def list_runs() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    dirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and (p / STATUS).exists()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]


def latest_run() -> str | None:
    runs = list_runs()
    return runs[0] if runs else None
