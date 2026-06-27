#!/usr/bin/env bash
# Wait for train_genreg.py to finish (adapter dir + 'saved adapter' log line), then run the A/B eval,
# the extractor, and the fluency check. Foreground inside its own background nohup so steps 2-4 complete
# unattended. Paths are derived from this script's location; runs on GPU 0.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# decouple-training/ root, derived from this script (gen-calibration/run_eval_when_ready.sh)
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CP=$BASE/controllable-posterior/core
GC=$BASE/gen-calibration
SE=$BASE/state-emission
ADAPTER=$CP/outputs/adapter_oracle_decoupled_genreg_s0
TRAINLOG=$GC/outputs/train_genreg.log

cd "$BASE"

echo "[watch] waiting for adapter + saved-line ..." >> "$GC/outputs/eval_pipeline.log"
# wait until the training log records the save AND the adapter weights exist
for i in $(seq 1 240); do   # up to ~120 min
  if grep -q "saved adapter" "$TRAINLOG" 2>/dev/null && \
     { [ -f "$ADAPTER/adapter_model.safetensors" ] || [ -f "$ADAPTER/adapter_model.bin" ]; }; then
    echo "[watch] adapter ready at $(date)" >> "$GC/outputs/eval_pipeline.log"
    break
  fi
  sleep 30
done

if ! { [ -f "$ADAPTER/adapter_model.safetensors" ] || [ -f "$ADAPTER/adapter_model.bin" ]; }; then
  echo "[watch] TIMEOUT: adapter never appeared" >> "$GC/outputs/eval_pipeline.log"
  exit 1
fi

echo "[eval] === A/B state-emission (decoupled vs decoupled_genreg), b1+b2 === $(date)" >> "$GC/outputs/eval_pipeline.log"
python "$SE/run_state_emission.py" \
  --cp_root "$CP" \
  --groups decoupled,decoupled_genreg \
  --regimes b1,b2 --ks 2,3,4,5 --per_cell 10 --n 16 --temps 1.0 \
  --temp_free 1.0 --llm_subset 40 --tag genreg \
  >> "$GC/outputs/eval_pipeline.log" 2>&1
echo "[eval] state-emission rc=$?" >> "$GC/outputs/eval_pipeline.log"

# copy the two diversity_*.json into gen-calibration/outputs for self-containment (eval writes to state-emission/outputs)
cp -f "$SE/outputs/diversity_decoupled.json" "$GC/outputs/" 2>/dev/null
cp -f "$SE/outputs/diversity_decoupled_genreg.json" "$GC/outputs/" 2>/dev/null

echo "[extract] === structured-commit A/B by j ===" >> "$GC/outputs/eval_pipeline.log"
python "$GC/extract_ab.py" "$SE/outputs" decoupled decoupled_genreg \
  >> "$GC/outputs/eval_pipeline.log" 2>&1
echo "[extract] rc=$?" >> "$GC/outputs/eval_pipeline.log"

echo "[fluency] === heldout perplexity + KL ===" >> "$GC/outputs/eval_pipeline.log"
python "$GC/fluency_check.py" >> "$GC/outputs/eval_pipeline.log" 2>&1
echo "[fluency] rc=$?" >> "$GC/outputs/eval_pipeline.log"

echo "[done] pipeline finished at $(date)" >> "$GC/outputs/eval_pipeline.log"
