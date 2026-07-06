#!/bin/bash
#SBATCH --job-name=bert_distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --output=distributed_output.log
#SBATCH --error=distributed_error.log

set -euo pipefail

source "$STORE/dl_venv/bin/activate"

export GPUS_PER_NODE=1
export MASTER_PORT=${MASTER_PORT:-29500}

cd "$SLURM_SUBMIT_DIR"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash ./launch_distributed.sh