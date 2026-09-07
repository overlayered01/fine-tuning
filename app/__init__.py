"""Gradio 학습 콘솔.

env 를 가장 먼저 import 한다. huggingface_hub·torch·gradio 는 import 시점에
캐시 경로를 확정해버리므로, 그 전에 환경변수가 잡혀 있어야 한다.
"""

from . import env as env  # noqa: F401  (import 부수효과로 캐시 경로가 고정된다)
