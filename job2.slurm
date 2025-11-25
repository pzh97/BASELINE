#!/bin/bash
#SBATCH --job-name=bert_ddp
#SBATCH --output=ddp_output.log
#SBATCH --error=ddp_error.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:2
#SBATCH -p short

# Activate environment
source $STORE/dl_venv/bin/activate

# Output directories
OUT_DIR=$STORE/bert_outputs/squad_ddp
LOG_DIR=$STORE/tensorboard_logs/bert_ddp
mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

# Number of GPUs per node
NUM_GPUS=2

# Launch training with torchrun (DDP)
srun torchrun --nproc_per_node=$NUM_GPUS ./baseline/run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUT_DIR \
  --logging_dir $LOG_DIR \
  --report_to tensorboard \
  --fp16 \
  --gradient_accumulation_steps 1 \
  --save_strategy=steps
