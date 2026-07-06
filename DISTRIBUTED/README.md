# Deliverable 2: DISTRIBUTED - Distributed implementation

## Objective

This folder contains the distributed version of the Question Answering training workflow used in the BASELINE deliverable. The goal of this deliverable is to parallelize the training code with native PyTorch distributed support, run it on multiple GPUs and nodes, and report execution times and final metrics.

The model used is `google-bert/bert-base-uncased` and the dataset is `squad` from Hugging Face Datasets.

## Chosen Distributed Strategy

The main implementation uses native PyTorch `torchrun` with **Distributed Data Parallel (DDP)**.

This was chosen because:

- it is the standard native PyTorch solution for multi-GPU and multi-node training
- it integrates cleanly with the Hugging Face `Trainer`
- it requires very small changes to the validated baseline code
- it is simpler and more stable than moving directly to a sharded strategy

The execution layout for the official run is:

- 2 nodes
- 2 A100 GPUs per node
- 1 process per GPU
- 4 training processes in total

Mixed precision is enabled with `--fp16`.

In DDP, each process owns one GPU and keeps a full copy of the model. Each process receives a different shard of the input mini-batch, performs its own forward and backward pass locally, and then synchronizes gradients with the other processes during backpropagation. After the gradients are averaged, every process applies the same optimizer update, so all model replicas stay identical.

For this deliverable, that means the training data is split across 4 workers in parallel while the model itself is replicated on all 4 GPUs. The main benefit of this approach is reduced wall-clock training time, since multiple mini-batches are processed at the same time. The main overhead comes from gradient synchronization across GPUs and across nodes.

## Additional Strategy Tested

An additional **FSDP** launcher is also included in this folder as an exploratory second strategy.

The FSDP configuration uses:

- `--fsdp "full_shard auto_wrap"`
- `--fsdp_transformer_layer_cls_to_wrap BertLayer`

FSDP differs from DDP in the way model states are distributed. Instead of storing a full copy of the model, gradients, and optimizer states on every GPU, FSDP shards those tensors across workers. This reduces per-GPU memory usage and can make larger models feasible. In exchange, it introduces more communication and more complex coordination, because parameters often need to be gathered before computation and resharded afterwards.

For the deliverable results, both DDP and FSDP now have completed runs, although DDP remains the faster configuration on this setup.

## Implementation Approach

The distributed folder reuses the existing baseline training script instead of duplicating the training logic.

`run_qa.py` in this folder is a thin wrapper that loads `BASELINE/run_qa.py`. This keeps the model code, preprocessing, and evaluation logic identical to the baseline and changes only the execution mode.

The distributed behavior comes from the launch scripts:

- `job.slurm`: SLURM job for the DDP run
- `launch_distributed.sh`: multi-node `torchrun` launcher for DDP
- `job_fsdp.slurm`: SLURM job for the FSDP run
- `launch_fsdp.sh`: multi-node `torchrun` launcher for FSDP
- `extract_metrics.sh`: helper script to extract timing and throughput values from a log file

## How the DDP Launch Works

`job.slurm` requests the official resources for the distributed run:

- 2 nodes
- 1 SLURM task per node
- 2 A100 GPUs per node
- 64 CPUs per task
- 64 GB RAM per node

Inside the allocated nodes, `srun` starts `launch_distributed.sh`. That script derives the distributed parameters from the SLURM environment and launches:

```bash
torchrun \
	--nnodes="$NNODES" \
	--nproc_per_node="$GPUS_PER_NODE" \
	--node_rank="$NODE_RANK" \
	--master_addr="$MASTER_ADDR" \
	--master_port="$MASTER_PORT"
```

The training configuration used by the launcher is:

- model: `google-bert/bert-base-uncased`
- dataset: `squad`
- train batch size per device: `4`
- eval batch size per device: `8`
- gradient accumulation steps: `1`
- learning rate: `3e-5`
- epochs: `2`
- max sequence length: `384`
- doc stride: `128`
- preprocessing workers: `8`
- precision: `fp16`

The effective global train batch size is:

$$
4\ \text{GPUs} \times 4\ \text{samples per GPU} \times 1\ \text{accumulation step} = 16
$$

## How to Run

### DDP run

```bash
sbatch job.slurm
```

### FSDP run

```bash
sbatch job_fsdp.slurm
```

### Extract metrics from a finished log

```bash
./extract_metrics.sh distributed_output.log
```

## Test and Measured Results

The DDP implementation was tested on the required distributed setup and completed successfully.

### Completed DDP run

Source logs:

- `distributed_output.log`
- `distributed_error.log`

Final reported training values:

- epochs: `2.0`
- world size: `4`
- measured wall-clock training time: `857.22 s`
- trainer `train_runtime`: `856.62 s`
- training samples: `88524`
- training throughput: `206.682 samples/s`
- training step throughput: `12.918 steps/s`
- final train loss: `1.0026`
- total FLOPs: `32315140 GF`

Final evaluation values:

- eval exact match: `81.1164`
- eval F1: `88.4441`
- eval runtime: `6.00 s`
- eval samples: `10784`
- eval throughput: `1795.442 samples/s`
- eval step throughput: `56.108 steps/s`

### Speedup relative to the baseline

Baseline training time from Deliverable 1:

- baseline wall-clock training time: `2828.00 s`

Distributed DDP training time from this deliverable:

- distributed wall-clock training time: `857.22 s`

The speedup is computed as:

$$
speedup = \frac{2828.00}{857.22} \approx 3.299
$$

So the distributed DDP implementation achieved:

- speedup: `3.30x`
- time reduction: `69.69%`

Because the DDP run used 4 GPUs in total, it is also useful to report the parallel efficiency:

$$
parallel\ efficiency = \frac{3.299}{4} \approx 0.8248
$$

This corresponds to an efficiency of about `82.48%`, which is reasonable for multi-node distributed training because some time is always lost to synchronization and communication overhead.

In shorter form, the successful distributed run finished training in about **14 minutes 17 seconds** on **4 GPUs across 2 nodes**, then completed evaluation with **EM = 81.12** and **F1 = 88.44**.

### Completed FSDP run

Source logs:

- `fsdp_output.log`
- `fsdp_error.log`

Final reported training values:

- epochs: `2.0`
- world size: `4`
- measured wall-clock training time: `1056.22 s`
- trainer `train_runtime`: `1051.63 s`
- training samples: `88524`
- training throughput: `168.356 samples/s`
- training step throughput: `10.523 steps/s`
- final train loss: `1.0085`
- total FLOPs: `8078785 GF`

Final evaluation values:

- eval exact match: `81.2204`
- eval F1: `88.5142`
- eval runtime: `10.72 s`
- eval samples: `10784`
- eval throughput: `1005.141 samples/s`
- eval step throughput: `31.411 steps/s`

Relative to the baseline, the completed FSDP run achieved:

- speedup: `2.68x`
- time reduction: `62.65%`

Relative to the completed DDP run, the FSDP runtime was:

- `1.23x` slower than DDP

This means FSDP did complete correctly after the optimizer change, but for this specific BERT-Base and SQuAD workload it did not outperform DDP in wall-clock time.

For completeness, the first FSDP attempt failed during the optimizer step with:

```text
RuntimeError: output with shape [] doesn't match the broadcast shape [1]
```

That failure is no longer the final state of the FSDP experiment, because the patched rerun completed successfully.

## Deliverable Summary

This deliverable satisfies the requirement to parallelize the baseline code using native PyTorch distributed training.

- Primary implementation: native PyTorch DDP with `torchrun`
- Additional explored strategy: FSDP
- Official tested distributed results: both DDP and FSDP completed successfully on 2 nodes with 2 A100 GPUs per node
- Reported measurements: wall-clock training time, trainer runtime, throughput, and final evaluation metrics for both strategies

The best wall-clock result in this deliverable is the DDP run, while the FSDP run serves as an additional completed distributed strategy for comparison.