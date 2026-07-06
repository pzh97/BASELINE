#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 LOG_FILE" >&2
  exit 1
fi

LOG_FILE=$1

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file not found: $LOG_FILE" >&2
  exit 1
fi

grep -E "Training time:|train_runtime|train_samples_per_second|train_steps_per_second|world_size|measured_train_wall_time_seconds" "$LOG_FILE" | tail -n 20