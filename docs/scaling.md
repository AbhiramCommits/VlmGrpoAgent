# Scaling notes (measured, not estimated)

Hardware: **mps**, model tier **tiny** (256,484,928 parameters), dtype **float32**.

## Rollout throughput

Measured with `make bench` on the test split, max 1 turns and 16 new tokens per turn, greedy-free sampling (T=0.9).

| batch size | rollouts | turns | wall (s) | rollouts/min | s/rollout | peak MPS/CUDA mem (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 3.58 | 16.74 | 3.58 | 1693.2 |
| 4 | 4 | 4 | 9.81 | 24.45 | 2.45 | 3186.5 |

## Real GRPO training step

| metric | value |
|---|---:|
| steps | 3 |
| group size | 4 |
| rollouts | 12 |
| rollouts/min | 1.59 |
| mean step seconds | 112.51 |
| peak memory (MB) | 5281.0 |
| wall clock (min) | 7.5 |
| skipped zero-variance groups | 1 |

## SFT step

150 optimizer steps over 400 examples (batch 1 x grad-accum 1), 25.3 min wall clock, final val loss 0.1342.

## Not measured on this machine (no CUDA)

The following commands are wired but were not executed here:

```bash
# single node, 2-8 GPUs, DeepSpeed ZeRO-2
accelerate launch --config_file configs/accel/ds_zero2.json --num_processes 4 \
  -m vlmgrpo.train_grpo --config configs/small.yaml

# FSDP full shard
accelerate launch --config_file configs/accel/fsdp.yaml --num_processes 4 \
  -m vlmgrpo.train_grpo --config configs/small.yaml
```

Rollout throughput on CUDA is expected to be dominated by generation; pass `--backend vllm` (or `rollout.backend: vllm`) to use the offline vLLM engine when it is installed. The agent records the effective backend in every evaluation JSON and falls back to HF generation with a warning when vLLM/CUDA is unavailable.
