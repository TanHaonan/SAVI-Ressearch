#!/usr/bin/env bash
# Download models + datasets for the PRM boundary-step calibration check.
# Idempotent: `hf download` resumes partial files and skips complete ones, so
# re-running is cheap. Most assets are already in the cache on this box.
#
# Usage:  bash download_assets.sh            # primary + control + substrate
#         WITH_CONTRAST=1 bash download_assets.sh   # also fetch contrast PRMs
set -euo pipefail

# hf-mirror is the working endpoint here; the public hub also resolves but the
# mirror is faster from this node. HF_HOME points at the shared cache.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# Optional accelerated transfer if the package is installed.
python -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1 || true

# Retry wrapper — the proxy occasionally drops the first HEAD; a retry fixes it.
retry() { local n=$1; shift; local i=1
  until "$@"; do
    [ "$i" -ge "$n" ] && { echo "FAILED after ${n}x: $*" >&2; return 1; }
    echo "retry ${i}/${n}: $*" >&2; i=$((i+1)); sleep 5
  done; }

dl_model()   { echo ">> model   $1"; retry 5 hf download "$1" >/dev/null; }
dl_dataset() { echo ">> dataset $1"; retry 5 hf download "$1" --repo-type dataset >/dev/null; }

# --- primary value under test: the PRM ---
dl_model   Qwen/Qwen2.5-Math-PRM-7B

# --- calibration control: LM-as-step-judge, size-matched to the 7B PRM ---
dl_model   Qwen/Qwen2.5-7B-Instruct

# --- labeled substrate: human-annotated first-error step ---
dl_dataset Qwen/ProcessBench

# --- optional contrast PRMs (opposite biases) and fine-grained subcategory set ---
if [ "${WITH_CONTRAST:-0}" = "1" ]; then
  dl_model   peiyi9979/math-shepherd-mistral-7b-prm          # on-policy MC PRM (already cached)
  dl_model   Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B         # negative-bias PRM
  # PRMBench dataset id — confirm before enabling (repo name varies by mirror):
  # dl_dataset hitsmy/PRMBench_Preview
fi

echo "DONE. Cache root: $HF_HOME/hub"
