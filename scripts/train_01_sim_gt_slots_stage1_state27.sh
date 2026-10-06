#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-evo1_hsr}"
set +u
conda activate "${CONDA_ENV}"
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EVO_DIR}"

RUN_NAME="${RUN_NAME:-hsr_sim_gt_slots_stage1_state27_bs16_h50_img448}"
SAVE_DIR="${SAVE_DIR:-${EVO_DIR}/checkpoints/${RUN_NAME}}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/home/mibo/study/Evo-1/Evo_1_flash/Evo_1/pretrained_ckpt/Evo1_LIBERO_STD}"
DATASET_CONFIG="${DATASET_CONFIG:-dataset/config_demo_hsr_tomato_randomrect_realmatch_mix80_state27.yaml}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/evo_mibo4_${RUN_NAME}}"
DS_CONFIG="${DS_CONFIG:-ds_config/zero2.json}"

LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
MAX_STEPS="${MAX_STEPS:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-6}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
HORIZON="${HORIZON:-50}"
MEMORY_SLOT_COUNT="${MEMORY_SLOT_COUNT:-4}"
SLOT_RELATION_MIN_DISTANCE="${SLOT_RELATION_MIN_DISTANCE:-1e-3}"

for required in "${PRETRAIN_CKPT}" "${DS_CONFIG}" "${DATASET_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] required path not found: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${SAVE_DIR}" "${CACHE_DIR}"
export SWANLAB_MODE="${SWANLAB_MODE:-disabled}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[mibo4-01-sim-gt-slots-stage1] dataset=${DATASET_CONFIG}"
echo "[mibo4-01-sim-gt-slots-stage1] save_dir=${SAVE_DIR}"
echo "[mibo4-01-sim-gt-slots-stage1] input=images+text+state27, state27=robot15+4*[dx,dy,valid]"
echo "[mibo4-01-sim-gt-slots-stage1] VLM frozen, action head + geometric memory trainable"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[mibo4-01-sim-gt-slots-stage1] DRY_RUN=1, exiting before accelerate launch."
  exit 0
fi

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --deepspeed_config_file "${DS_CONFIG}" \
  scripts/train.py \
  --run_name "${RUN_NAME}" \
  --action_head flowmatching \
  --use_augmentation \
  --vision_masked \
  --lr "${LR}" \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size "${BATCH_SIZE}" \
  --image_size "${IMAGE_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --ckpt_interval "${CKPT_INTERVAL}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon "${HORIZON}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor 2 \
  --video_backend av \
  --cache_dir "${CACHE_DIR}" \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path "${DATASET_CONFIG}" \
  --per_action_dim 9 \
  --state_dim 27 \
  --state_encoder_type geometric_memory \
  --robot_state_dim 15 \
  --slot_count 4 \
  --slot_dim 3 \
  --slot_position_dim 2 \
  --memory_slot_count "${MEMORY_SLOT_COUNT}" \
  --slot_relation_min_distance "${SLOT_RELATION_MIN_DISTANCE}" \
  --save_dir "${SAVE_DIR}" \
  --resume \
  --resume_pretrain \
  --resume_path "${PRETRAIN_CKPT}"
