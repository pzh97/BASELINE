#!/bin/bash
#SBATCH --job-name=bert_baseline          # Name of the job
#SBATCH --output=baseline_output.log      # Standard output log
#SBATCH --error=baseline_error.log        # Standard error log
#SBATCH --time=01:00:00                   # Time limit (hh:mm:ss)
#SBATCH --mem=16G                         # Total memory
#SBATCH --cpus-per-task=32                # Number of CPU cores per task
#SBATCH --gres=gpu:a100:1                 # Request 1 A100 GPU
#SBATCH -p short                          # Partition to submit to

# Load required modules
# module load python/3.10
# module load cuda/11.7

# Activate virtual environment
source $STORE/dl_venv/bin/activate

# Output and logging paths
OUT_DIR=$STORE/bert_outputs/squad_baseline
LOG_DIR=$STORE/tensorboard_logs/bert_baseline

mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

# Run the training script
srun python ./baseline/run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUT_DIR \
  --logging_dir $LOG_DIR \
  --report_to tensorboard
