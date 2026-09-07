"""학습 subprocess 생성 / 중단 / 상태 조회.

PLAN-WEBUI.md §1.1 — 학습은 Gradio 프로세스 안에서 돌리지 않는다.
UI 블로킹 · 중단 불가 · 크래시 전파를 한 번에 피하기 위해 별도 프로세스로 띄운다.

중단은 §3의 3단계:
  1) STOP 파일 생성 -> 학습 쪽 콜백이 감지해 정상 종료 (체크포인트 보존)
  2) 유예 시간 대기
  3) 그래도 안 죽으면 강제 종료
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from . import state

PROJECT_ROOT = state.PROJECT_ROOT

GRACE_SEC = 30.0

# 같은 프로젝트 폴더를 Windows 와 WSL 양쪽에서 쓴다. 경로만 보고 고르면
# WSL 에서 .venv/Scripts/python.exe 를 실행하려다 Exec format error 가 난다.
VENV_CANDIDATES = (
    [PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"] if os.name == "nt"
    else [PROJECT_ROOT / ".venv-wsl" / "bin" / "python",
          PROJECT_ROOT / ".venv" / "bin" / "python"]
)


def _python() -> str:
    """이 플랫폼에서 실제로 실행 가능한 venv 파이썬을 고른다."""
    for c in VENV_CANDIDATES:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return sys.executable


class TrainRunner:
    """한 번에 하나의 학습만 관리한다. GPU가 1장이므로 동시 실행은 무의미하다."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._run: str | None = None

    # ── 조회 ──────────────────────────────────────────────────────────

    @property
    def current_run(self) -> str | None:
        return self._run

    def is_alive(self) -> bool:
        """이 UI 프로세스가 띄운 자식이 살아있는지."""
        return self._proc is not None and self._proc.poll() is None

    def busy_run(self) -> str | None:
        """지금 학습 중인 run 이름. UI 재시작 후에도 파일을 보고 복구한다."""
        if self.is_alive() and self._run:
            return self._run
        for name in state.list_runs():
            st = state.read_status(state.run_dir(name))
            if st["state"] in (state.RUNNING, state.STOPPING) and not state.is_stale(st):
                return name
        return None

    # ── 실행 ──────────────────────────────────────────────────────────

    def start(self, script: Path, run_name: str, config_path: Path) -> tuple[bool, str]:
        busy = self.busy_run()
        if busy:
            return False, f"이미 학습이 실행 중입니다 — {busy}"

        d = state.ensure_run_dir(run_name)
        state.clear_stop(d)
        # 이전 실행 흔적을 지운다. 차트가 과거 run 의 loss 와 섞이면 안 된다.
        for f in (state.METRICS, state.LOG):
            try:
                (d / f).unlink()
            except OSError:
                pass

        state.write_status(
            d, state=state.RUNNING, step=0, max_steps=0,
            started_at=time.time(), run=run_name, error=None,
        )

        log_f = (d / state.LOG).open("a", encoding="utf-8", buffering=1)

        # Windows 기본 콘솔 인코딩은 cp949 라, 자식이 한글이나 em-dash 를 print 하면
        # UnicodeEncodeError 로 학습이 통째로 죽는다. utf-8 을 강제한다.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

        # -u: 자식의 stdout 버퍼링을 끈다. 켜두면 로그가 뭉텅이로 늦게 도착한다.
        self._proc = subprocess.Popen(
            [_python(), "-u", str(script), "--config", str(config_path), "--run", run_name],
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._run = run_name
        state.patch_status(d, pid=self._proc.pid)
        return True, f"학습 시작 — {run_name} (pid {self._proc.pid})"

    # ── 중단 ──────────────────────────────────────────────────────────

    def stop(self, run_name: str | None = None) -> tuple[bool, str]:
        run_name = run_name or self.busy_run()
        if not run_name:
            return False, "실행 중인 학습이 없습니다."

        d = state.run_dir(run_name)
        state.request_stop(d)
        state.patch_status(d, state=state.STOPPING)
        return True, "중단을 요청했습니다. 체크포인트 저장 후 종료됩니다."

    def force_kill(self, run_name: str | None = None) -> tuple[bool, str]:
        """3단계 — 유예 시간이 지나도 안 죽을 때의 백업 경로."""
        run_name = run_name or self.busy_run()
        if not run_name:
            return False, "실행 중인 학습이 없습니다."
        d = state.run_dir(run_name)

        if self.is_alive() and self._proc is not None:
            self._proc.terminate()
            self._proc = None
        else:
            # UI 재시작 후라 Popen 핸들이 없다 — status.json 의 pid 로 잡는다.
            pid = state.read_status(d).get("pid")
            if pid:
                try:
                    psutil.Process(int(pid)).terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                    pass

        state.patch_status(d, state=state.FAILED, error="사용자가 강제 종료했습니다.")
        state.clear_stop(d)
        return True, "강제 종료했습니다."

    def reap(self) -> None:
        """죽었는데 status 가 running 으로 남은 경우를 정리한다.

        학습 프로세스가 status 를 갱신하지 못하고 죽는 경우(세그폴트, OOM 킬)가 있다.
        """
        for name in state.list_runs():
            d = state.run_dir(name)
            st = state.read_status(d)
            if st["state"] in (state.RUNNING, state.STOPPING) and state.is_stale(st):
                state.patch_status(
                    d, state=state.FAILED,
                    error="프로세스 응답이 끊겼습니다. train.log 를 확인하세요.",
                )


RUNNER = TrainRunner()
