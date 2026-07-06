#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

NNODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${SLURM_NODEID:-0}

STORE_ROOT=${STORE:-$PROJECT_ROOT}
OUT_DIR=${OUT_DIR:-$STORE_ROOT/bert_outputs/squad_distributed_a100}
LOG_DIR=${LOG_DIR:-$STORE_ROOT/tensorboard_logs/bert_distributed_a100}
CACHE_DIR=${CACHE_DIR:-$STORE_ROOT/huggingface_cache}

TRAIN_BATCH=${TRAIN_BATCH:-4}
EVAL_BATCH=${EVAL_BATCH:-8}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-384}
DOC_STRIDE=${DOC_STRIDE:-128}
PREPROC_WORKERS=${PREPROC_WORKERS:-8}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CACHE_DIR"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$SCRIPT_DIR/run_qa.py" \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size "$TRAIN_BATCH" \
  --per_device_eval_batch_size "$EVAL_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_EPOCHS" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --doc_stride "$DOC_STRIDE" \
  --preprocessing_num_workers "$PREPROC_WORKERS" \
  --output_dir "$OUT_DIR" \
  --logging_dir "$LOG_DIR" \
  --save_strategy steps \
  --logging_strategy steps \
  --logging_steps 100 \
  --report_to tensorboard \
  --cache_dir "$CACHE_DIR"