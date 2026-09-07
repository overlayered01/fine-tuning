# 셸에서 uv / huggingface-cli / python 을 직접 부를 때 캐시를 프로젝트에 묶는다.
#   . .\scripts\env.ps1
$FtRoot  = Split-Path -Parent $PSScriptRoot
$FtCache = Join-Path $FtRoot ".cache"

$env:HF_HOME           = Join-Path $FtCache "hf"
$env:HF_HUB_CACHE      = Join-Path $FtCache "hf\hub"
$env:HF_XET_CACHE      = Join-Path $FtCache "hf\xet"
$env:HF_DATASETS_CACHE = Join-Path $FtCache "hf\datasets"
$env:TORCH_HOME        = Join-Path $FtCache "torch"
$env:TRITON_CACHE_DIR  = Join-Path $FtCache "triton"
$env:GRADIO_TEMP_DIR   = Join-Path $FtCache "gradio"
$env:MPLCONFIGDIR      = Join-Path $FtCache "mpl"
$env:XDG_CACHE_HOME    = $FtCache
$env:UV_CACHE_DIR      = Join-Path $FtCache "uv"

foreach ($p in @($env:HF_HOME, $env:HF_XET_CACHE, $env:TORCH_HOME, $env:TRITON_CACHE_DIR,
                 $env:GRADIO_TEMP_DIR, $env:MPLCONFIGDIR, $env:UV_CACHE_DIR)) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}
Write-Host "[env] 캐시 -> $FtCache"
