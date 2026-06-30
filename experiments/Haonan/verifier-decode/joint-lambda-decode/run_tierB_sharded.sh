#!/usr/bin/env bash
# Tier-B decisive run, sharded across GPUs 4-7 (PLAN3). Each shard = one backend profile
# over a k-group; the two k=6 cells (the long pole) get a GPU each. After all shards
# finish, merge_shards.py assembles the final results.json + synthesis.
#
# Usage: bash run_tierB_sharded.sh [N] [TAG]
set -u
cd "$(dirname "$0")"

N="${1:-16}"
TAG="${2:-tierB}"
KEFF_N=8
BOM_CAP=200
K=8 ; NS=8 ; TAU=1.0 ; SEED=0 ; NBOOT=10000
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# Repo-relative path to the emission line's countdown-decode adapters. Override REPO_ROOT
# (or EMIT directly) if the checkout lives elsewhere.
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
EMIT=$REPO_ROOT/contrib-savi/experiments/Haonan/decouple-training/countdown-decode/outputs
DEC=$EMIT/adapter_decoupled_s0
COU=$EMIT/adapter_coupled_s0
mkdir -p outputs

common="--tier B --device cuda --K $K --N $NS --tau $TAU --seed $SEED --n $N \
  --bom_max_rollouts $BOM_CAP --keff_n $KEFF_N --n_boot $NBOOT"

echo "launching 4 shards (N=$N) across GPUs 4-7..."
CUDA_VISIBLE_DEVICES=4 nohup python3 -u run_joint.py $common --only_profile decoupled \
  --adapter_decoupled $DEC --ks 6 --out outputs/${TAG}_dec_k6.json \
  > outputs/${TAG}_dec_k6.log 2>&1 &
echo "  GPU4: decoupled k=6  PID $!"
CUDA_VISIBLE_DEVICES=5 nohup python3 -u run_joint.py $common --only_profile coupled \
  --adapter_coupled $COU --ks 6 --out outputs/${TAG}_cou_k6.json \
  > outputs/${TAG}_cou_k6.log 2>&1 &
echo "  GPU5: coupled   k=6  PID $!"
CUDA_VISIBLE_DEVICES=6 nohup python3 -u run_joint.py $common --only_profile decoupled \
  --adapter_decoupled $DEC --ks 4,5 --out outputs/${TAG}_dec_k45.json \
  > outputs/${TAG}_dec_k45.log 2>&1 &
echo "  GPU6: decoupled k=4,5 PID $!"
CUDA_VISIBLE_DEVICES=7 nohup python3 -u run_joint.py $common --only_profile coupled \
  --adapter_coupled $COU --ks 4,5 --out outputs/${TAG}_cou_k45.json \
  > outputs/${TAG}_cou_k45.log 2>&1 &
echo "  GPU7: coupled   k=4,5 PID $!"
echo "shards launched. When all 4 JSONs exist, run:"
echo "  python3 merge_shards.py --shards 'outputs/${TAG}_*.json' --out outputs/results.json"
