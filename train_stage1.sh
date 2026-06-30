#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}"
: "${VISION_TOWER:?Set VISION_TOWER}"
: "${VISION_PRETRAINED:?Set VISION_PRETRAINED}"
: "${DATASET_DIR:?Set DATASET_DIR}"
: "${TRAIN_JSON:?Set TRAIN_JSON}"
: "${EVAL_JSON:?Set EVAL_JSON}"
: "${GPU_IDS:?Set GPU_IDS}"
: "${EPOCHS:?Set EPOCHS}"
: "${STEPS_PER_EPOCH:?Set STEPS_PER_EPOCH}"
: "${GRAD_ACCUMULATION_STEPS:?Set GRAD_ACCUMULATION_STEPS}"
: "${BATCH_SIZE:?Set BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASTER_PORT="${MASTER_PORT:-24998}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
EXP_NAME="${EXP_NAME:-mediround_stage1}"

cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" deepspeed --master_port="${MASTER_PORT}" train_ds.py \
  --stage 1 \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --dataset_dir "${DATASET_DIR}" \
  --train_json "${TRAIN_JSON}" \
  --eval_json "${EVAL_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --exp_name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --conv_type "${CONV_TYPE:-mistral_instruct}" \
  "$@"
