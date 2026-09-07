# 셸에서 uv / huggingface-cli / python 을 직접 부를 때 캐시를 프로젝트에 묶는다.
#   source scripts/env.sh
# 파이썬 진입점은 app/env.py 가 알아서 처리하므로 이건 CLI 도구용이다.
FT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FT_CACHE="$FT_ROOT/.cache"

export HF_HOME="$FT_CACHE/hf"
export HF_HUB_CACHE="$FT_CACHE/hf/hub"
export HF_XET_CACHE="$FT_CACHE/hf/xet"
export HF_DATASETS_CACHE="$FT_CACHE/hf/datasets"
export TORCH_HOME="$FT_CACHE/torch"
export TRITON_CACHE_DIR="$FT_CACHE/triton"
export GRADIO_TEMP_DIR="$FT_CACHE/gradio"
export MPLCONFIGDIR="$FT_CACHE/mpl"
export XDG_CACHE_HOME="$FT_CACHE"
export UV_CACHE_DIR="$FT_CACHE/uv"

mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"          "$GRADIO_TEMP_DIR" "$MPLCONFIGDIR" "$UV_CACHE_DIR"

# ── C 컴파일러 (triton JIT 용) ────────────────────────────────────────
# WSL 에 gcc 가 없고 apt 는 sudo 가 필요하다. PyPI ziglang 을 대신 쓴다.
if [ -x "$FT_ROOT/.venv-wsl/bin/python" ]; then
  export FT_PYTHON="$FT_ROOT/.venv-wsl/bin/python"
  if "$FT_PYTHON" -c "import ziglang" 2>/dev/null; then
    export CC="$FT_ROOT/.cache/bin/zig-cc"
    export CXX="$CC"
    export ZIG_GLOBAL_CACHE_DIR="$FT_CACHE/zig"
    export ZIG_LOCAL_CACHE_DIR="$FT_CACHE/zig"
    export FT_LIBDIR="$FT_CACHE/lib"
    mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$FT_LIBDIR"
    # triton 은 -l:libcuda.so.1 형식을 쓰는데 zig 의 lld 가 이를 못 찾는다.
    # zig-cc 래퍼가 -lcuda 로 바꾸므로, 여기에 맞는 .so 링크를 만들어둔다.
    for so in /usr/lib/wsl/lib/*.so.1; do
      [ -e "$so" ] || continue
      base=$(basename "$so" .so.1)
      [ -e "$FT_LIBDIR/$base.so" ] || ln -sf "$so" "$FT_LIBDIR/$base.so"
    done
  fi
fi

echo "[env] 캐시 -> $FT_CACHE"
