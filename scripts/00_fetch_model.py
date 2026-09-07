"""Qwen3-8B 가중치 사전 다운로드.

models/ 아래 프로젝트 로컬로 받는다. WSL2 로 전환해도 /mnt/d 로 그대로 접근할 수 있어
다시 받을 필요가 없다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# huggingface_hub 보다 먼저 — import 시점에 캐시 경로가 확정된다.
from app import env  # noqa: E402,F401

# hf_transfer 는 폐기됨 — 지금은 Xet 고성능 전송이 대체한다.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download  # noqa: E402

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
DEST = Path(__file__).resolve().parent.parent / "models" / REPO.split("/")[-1]

print(f"[fetch] {REPO} -> {DEST}", flush=True)
path = snapshot_download(
    repo_id=REPO,
    local_dir=str(DEST),
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"],
    max_workers=4,
)
total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
print(f"[fetch] 완료 — {total / 1e9:.1f} GB @ {path}", flush=True)
