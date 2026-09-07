"""캐시 경로를 프로젝트 안으로 고정한다.

기본값대로 두면 라이브러리들이 사용자 프로필(AppData, ~/.cache)에 흩어져 쓴다.
그러면 프로젝트 폴더만 옮기거나 백업해도 환경이 따라오지 않고,
다른 프로젝트와 캐시를 공유하다 버전이 꼬인다.

**반드시 huggingface_hub / torch / gradio 보다 먼저 실행되어야 한다.**
그래서 셸 스크립트가 아니라 app/__init__.py 최상단에서 import 한다.
라이브러리들이 import 시점에 경로를 확정해버리기 때문이다.

이미 설정된 전역 변수(예: HF_HOME=D:\\models\\hf)도 덮어쓴다.
전역 캐시를 쓰고 싶으면 FT_USE_GLOBAL_CACHE=1 로 이 동작을 끌 수 있다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / ".cache"

# 환경변수 이름 -> .cache 아래 하위 경로
LAYOUT: dict[str, str] = {
    # HuggingFace — HF_HOME 이 hub/datasets 를 함께 덮지만, xet 캐시는
    # HF_HOME 을 따라가지 않으므로 반드시 따로 지정해야 한다.
    "HF_HOME": "hf",
    "HF_HUB_CACHE": "hf/hub",
    "HF_XET_CACHE": "hf/xet",
    "HF_DATASETS_CACHE": "hf/datasets",
    # PyTorch
    "TORCH_HOME": "torch",
    # WSL 에서 torch.compile / bitsandbytes 가 쓰는 커널 캐시
    "TRITON_CACHE_DIR": "triton",
    # Gradio 업로드·플롯 임시 파일 (기본은 시스템 temp)
    "GRADIO_TEMP_DIR": "gradio",
    # matplotlib 폰트 캐시 (기본은 ~/.matplotlib)
    "MPLCONFIGDIR": "mpl",
    # Linux 계열 라이브러리 다수가 참조하는 캐시 루트
    "XDG_CACHE_HOME": ".",
    # uv 패키지 캐시 — 셸에서 uv 를 부를 때 쓰인다
    "UV_CACHE_DIR": "uv",
}

_applied = False


def apply(create: bool = True) -> dict[str, str]:
    """캐시 환경변수를 프로젝트 로컬로 고정하고, 적용된 경로를 돌려준다."""
    global _applied
    if os.environ.get("FT_USE_GLOBAL_CACHE") == "1":
        return {}

    resolved: dict[str, str] = {}
    for var, rel in LAYOUT.items():
        path = (CACHE_ROOT / rel).resolve() if rel != "." else CACHE_ROOT.resolve()
        os.environ[var] = str(path)
        resolved[var] = str(path)
        if create:
            path.mkdir(parents=True, exist_ok=True)

    _applied = True
    return resolved


def report() -> str:
    lines = [f"프로젝트 캐시 루트: {CACHE_ROOT}"]
    for var in LAYOUT:
        lines.append(f"  {var:<20} = {os.environ.get(var, '(미설정)')}")
    return "\n".join(lines)


def setup_compiler() -> str | None:
    """triton JIT 에 필요한 C 컴파일러를 잡아준다 (POSIX 전용).

    WSL Ubuntu 에는 gcc 가 없고 apt 는 sudo 를 요구한다. PyPI ziglang 을
    컴파일러로 쓰면 sudo 없이 해결되고 툴체인도 프로젝트 안에 남는다.
    셸에서 env.sh 를 source 하지 않고 UI 를 띄워도 동작하도록 여기서도 설정한다.
    """
    if os.name == "nt" or os.environ.get("CC"):
        return os.environ.get("CC")

    wrapper = CACHE_ROOT / "bin" / "zig-cc"
    if not wrapper.exists():
        return None

    libdir = CACHE_ROOT / "lib"
    libdir.mkdir(parents=True, exist_ok=True)
    # triton 의 -l:libfoo.so.1 을 zig-cc 가 -lfoo 로 바꾸므로 .so 링크가 필요하다
    wsl_lib = Path("/usr/lib/wsl/lib")
    if wsl_lib.is_dir():
        for so in wsl_lib.glob("*.so.1"):
            link = libdir / (so.name[:-len(".so.1")] + ".so")
            if not link.exists():
                try:
                    link.symlink_to(so)
                except OSError:
                    pass

    os.environ["CC"] = os.environ["CXX"] = str(wrapper)
    os.environ["FT_LIBDIR"] = str(libdir)
    os.environ.setdefault("FT_PYTHON", sys.executable)
    zig_cache = CACHE_ROOT / "zig"
    zig_cache.mkdir(parents=True, exist_ok=True)
    os.environ["ZIG_GLOBAL_CACHE_DIR"] = os.environ["ZIG_LOCAL_CACHE_DIR"] = str(zig_cache)
    return str(wrapper)


# import 만으로 적용된다. 진입점이 순서를 신경 쓰지 않아도 되게 한다.
apply()
setup_compiler()
