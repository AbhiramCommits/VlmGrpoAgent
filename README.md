# VLM-GRPO

Verifiable-reward reinforcement learning (**RLVR** via **GRPO**) for **multimodal tool-using agents**.
We post-train a small open vision-language model so that it can look at a chart, document table,
synthetic screenshot or 3-frame video strip, plan, make at least one **validated tool call**
(sandboxed Python, retrieval, or UI actions), and produce a final answer that is checked
**programmatically**. We compare BASE vs SFT-cold-start vs GRPO on pass@1, pass@4, multi-step
task-completion rate, tool-call schema validity and perception recall, then attribute every failed
trajectory to a single step-level bucket so PERCEPTION failures are separated from PLANNING and
TOOL-USE failures.

There is **no learned reward model**: every reward component is a pure function of the trajectory
and the programmatically generated ground truth.

---

## 1. What it does

* Generates a deterministic multimodal suite (seed -> identical tasks and PNGs) with four families:
  `chart_agg`, `doc_table`, `screenshot_ui`, `multi_frame`. No internet dataset download is involved.
* Runs a multi-turn ReAct agent over a VLM: `<think>...</think>` then either one
  `<tool>{"name": ..., "args": {...}}</tool>` call or `<answer>...</answer>`.
* Executes tools in a sandbox: `python_exec` (subprocess, import allowlist, restricted builtins,
  audit-hook network block, throw-away cwd, 2 s timeout), `retrieve` (over the task's own rendered
  text spans), `click(element_id)` and `read_field(name)` (with a click-before-read constraint).
* Scores rollouts with a composite **verifiable** reward and trains with:
  * **SFT cold start**: LoRA on the LM attention+MLP projections, vision tower frozen,
    assistant-span-only loss, gold trajectories executed through the real sandbox.
  * **GRPO / RLVR**: groups of G rollouts, group-relative advantages, PPO-style ratio clipping,
    KL penalty to the frozen SFT reference, multi-turn credit assignment.
* Evaluates {BASE, SFT, GRPO} x {test, test_hard}, writes one canonical `results/results.json`,
  attributes failures, and regenerates every README number from that file.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────┐
                    │ deterministic task generator (seed)      │
                    │ chart_agg | doc_table | screenshot_ui    │
                    │ multi_frame            -> PNG + question │
                    └───────────────────┬──────────────────────┘
                                        │  image + prompt (no numbers in text)
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ multi-turn ReAct agent over a VLM (LoRA policy)              │
        │   <think> plan </think>                                      │
        │   <tool>{"name": ..., "args": {...}}</tool>  |  <answer>     │
        └─────────────┬──────────────────────────────────┬─────────────┘
                      │ tool call                        │ final answer
                      ▼                                  ▼
        ┌───────────────────────────┐        ┌────────────────────────────┐
        │ tool sandbox              │        │ programmatic verifiers     │
        │ python_exec (2s, allow-   │        │ verify_answer (tol/string) │
        │   list, no network)       │        │ verify_tool_call_schema    │
        │ retrieve / click /        │        │ verify_tool_trace (order)  │
        │   read_field              │        │ perception_recall          │
        └─────────────┬─────────────┘        └─────────────┬──────────────┘
                      └───────────────┬────────────────────┘
                                      ▼
                         ┌─────────────────────────────┐
                         │ verifiable reward stack     │  (no learned RM)
                         │ answer + schema + trace     │
                         │ + format - length penalty   │
                         └──────────────┬──────────────┘
                                        │
                   ┌────────────────────┴─────────────────────┐
                   ▼                                          ▼
        ┌─────────────────────┐                   ┌────────────────────────┐
        │ SFT cold start      │  runs/<id>/sft    │ GRPO / RLVR            │
        │ LoRA, gold traces   │──────────────────▶│ group rollouts, clip,  │
        └─────────────────────┘                   │ KL to frozen SFT ref   │
                                                  └───────────┬────────────┘
                                                              ▼
                                          ┌────────────────────────────────────┐
                                          │ evaluation matrix + failure        │
                                          │ attribution -> results/results.json│
                                          │ -> README / RESULTS.md (generated) │
                                          └────────────────────────────────────┘
```

### File map

| path | role |
|---|---|
| `vlmgrpo/config.py` | dataclasses `TaskConfig`, `RolloutConfig`, `SFTConfig`, `GRPOConfig`, `EvalConfig`, `RewardConfig`, `RunConfig`; YAML loader |
| `vlmgrpo/registry.py` | device detection, model-tier auto-selection, SmolVLM/Qwen2-VL/Qwen2.5-VL loaders, offline `stub` VLM |
| `vlmgrpo/tracking.py` | JSONL tracker (always) + W&B mirror only when `WANDB_API_KEY` is set |
| `vlmgrpo/utils.py` | seeding, timers, git sha, JSON/JSONL helpers, checkpoint registry |
| `vlmgrpo/tasks/families.py` | four deterministic task families, PNG rendering, `TaskItem` |
| `vlmgrpo/tasks/generate.py` | suite generation, splits, `data/suite_stats.md` |
| `vlmgrpo/verify.py` | answer / schema / trace / perception verifiers (pure functions) |
| `vlmgrpo/tools.py` | tool registry + sandboxed executors |
| `vlmgrpo/prompts.py` | system prompt, chat rendering, observation format |
| `vlmgrpo/agent.py` | batched multi-turn rollouts, tolerant action parser, `Trajectory` |
| `vlmgrpo/rewards.py` | composite verifiable reward + breakdown |
| `vlmgrpo/evaluate.py` | pass@1 / pass@k / completion / schema / recall + per-family, writes eval JSON + trajectories |
| `vlmgrpo/data/build_sft.py` | gold trajectories executed through the real sandbox -> SFT JSONL |
| `vlmgrpo/train_sft.py` | LoRA SFT cold start |
| `vlmgrpo/grpo.py` | advantages, PPO clipping, KL, token log-probs (from scratch) |
| `vlmgrpo/train_grpo.py` | online GRPO loop, validation, checkpointing, metrics |
| `vlmgrpo/train_grpo_trl.py` | optional TRL `GRPOTrainer` route (import-guarded) |
| `vlmgrpo/analysis/compare.py` | full evaluation matrix -> `results/results.json` |
| `vlmgrpo/analysis/failures.py` | one-bucket-per-failure attribution + examples |
| `vlmgrpo/analysis/plots.py` | reward curve, completion, failure mix, per-family, reward components |
| `scripts/make_results_table.py` | regenerates `results/RESULTS.md` + README tables from `results.json` |
| `scripts/benchmark_rollouts.py` | measured rollout throughput -> `docs/scaling.md` |

### Reward definition

```
r = w_answer  * answer_correct          # verify_answer(pred, gold, tol=0.01)
  + w_tool    * tool_schema_valid_rate   # valid calls / attempted calls
  + w_trace   * required_tool_used       # ordered required-tool sequence executed
  + w_format  * format_ok               # <think> + parseable <answer>
  - w_length  * length_penalty          # min(1, assistant_chars / length_budget)
```

Exact weights from `configs/tiny.yaml` (also recorded in every `results.json`):

| weight | value |
|---|---:|
| `w_answer` | 1.0 |
| `w_tool` | 0.2 |
| `w_trace` | 0.3 |
| `w_format` | 0.1 |
| `w_length` | 0.05 |
| `length_budget` (chars) | 1600 |

Every component is logged separately during GRPO and can be ablated from the YAML.

---

## 3. How to run

```bash
# 1) environment (Python 3.10+; CPU, Apple MPS and CUDA all supported)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt      # or: make setup

# 2) deterministic task suite + PNGs + data/suite_stats.md
make data

# 3) BASE zero-shot on val / test / test_hard
make eval-base

# 4) LoRA SFT cold start on gold trajectories (builds SFT data first)
make sft

# 5) GRPO / RLVR from the SFT checkpoint
make grpo

# 6) full {BASE,SFT,GRPO} x {test,test_hard} matrix -> results/results.json
make eval-all

# 7) regenerate README tables + results/RESULTS.md from results.json
make report

# offline smoke test (stub model, no downloads) and tests
make smoke
make test
make lint
```

### Model tiers and device auto-selection

| tier | model | params | device |
|---|---|---|---|
| `stub` | locally initialised VLM (no download) | ~1.9M | any (tests/CI) |
| `tiny` | `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M | CPU / MPS / CUDA |
| `small` | `Qwen/Qwen2-VL-2B-Instruct` | 2B | CUDA >= 8 GB |
| `base` | `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | CUDA >= 14 GB |

`--model-tier` (or `model_tier:` in YAML) overrides auto-selection. `device: auto` resolves
CUDA -> MPS -> CPU; `dtype: auto` resolves bf16/fp16 on CUDA and fp32 on MPS/CPU. The chosen tier,
device, dtype, parameter count and wall-clock are written into every evaluation JSON and into
`results/results.json`.

On the build machine (Apple M3, 16 GB) the tiny tier is selected and SmolVLM's 4x4 image splitting is
disabled: 13 crops of 512x512 produce a 13k-token vision sequence whose attention alone needs roughly
11 GB per sample on MPS. The single-view 512px image keeps the synthetic text legible and fits the
machine; this is recorded in the environment block and is the reason the results are **tiny-tier**
results.

### Rollout backend

`rollout.backend: hf` (default) uses `model.generate`. `--backend vllm` / `rollout.backend: vllm`
uses an offline vLLM engine when `vllm` is installed **and** CUDA is available. If vLLM cannot
initialise, the agent prints a warning, falls back to HF generation, and the effective backend is
recorded in the evaluation JSON (`environment.backend`).

### Multi-GPU: accelerate / DeepSpeed / FSDP

The TRL route runs under `accelerate`:

```bash
# DeepSpeed ZeRO-2, 4 GPUs
accelerate launch --config_file configs/accel/accelerate_ds.yaml --num_processes 4 \
  -m vlmgrpo.train_grpo_trl --config configs/small.yaml

# FSDP full shard, 4 GPUs
accelerate launch --config_file configs/accel/fsdp.yaml --num_processes 4 \
  -m vlmgrpo.train_grpo_trl --config configs/small.yaml

# the custom multi-turn loop is single-device; select a device with --devices
python -m vlmgrpo.train_grpo --config configs/tiny.yaml --devices 0
```

Measured rollout throughput, step time and peak memory for the configurations actually run are in
[`docs/scaling.md`](docs/scaling.md).

---

## 4. Results

Everything below is generated by `scripts/make_results_table.py` from `results/results.json`; no
number is hand-typed. The suite is synthetic, the model is small and the evaluation sample is
limited - see the limitations block for what that does and does not support.

<!-- RESULTS_START -->
## Environment

model tier **tiny** (256,484,928 params), device **mps**, dtype **float32**, seed **1234**, SFT steps **150**, GRPO steps **3**, train wall **0.55 h**, eval wall **0.98 h**, total **1.52 device-hours**.

## Headline metrics

| split | checkpoint | pass@1 | pass@4 | task completion | tool schema valid | mean turns | perception recall |
|---|---|---:|---:|---:|---:|---:|---:|
| test | BASE | 0.000 | 0.000 | 0.000 | 0.000 | 3.88 | 0.101 |
| test | SFT | 0.000 | 0.000 | 0.000 | 0.500 | 3.06 | 0.247 |
| test | GRPO | 0.000 | 0.000 | 0.000 | 0.500 | 2.75 | 0.247 |
| test_hard | BASE | 0.000 | 0.000 | 0.000 | 0.000 | 4.00 | 0.063 |
| test_hard | SFT | 0.000 | 0.000 | 0.000 | 0.118 | 2.81 | 0.266 |
| test_hard | GRPO | 0.000 | 0.000 | 0.000 | 0.154 | 2.62 | 0.289 |

## Deltas vs BASE and vs SFT

Absolute deltas (relative in parentheses). `vs BASE` shows the full effect of post-training; `vs SFT` isolates the RLVR step.

| split | checkpoint | reference | pass@1 | completion | schema valid | perception recall |
|---|---|---|---:|---:|---:|---:|
| test | BASE | SFT | +0.000 (n/a) | +0.000 (n/a) | -0.500 (-100.0%) | -0.146 (-59.3%) |
| test | SFT | BASE | +0.000 (n/a) | +0.000 (n/a) | +0.500 (n/a) | +0.146 (+145.7%) |
| test | GRPO | BASE | +0.000 (n/a) | +0.000 (n/a) | +0.500 (n/a) | +0.146 (+145.7%) |
| test | GRPO | SFT | +0.000 (n/a) | +0.000 (n/a) | +0.000 (+0.0%) | +0.000 (+0.0%) |
| test_hard | BASE | SFT | +0.000 (n/a) | +0.000 (n/a) | -0.118 (-100.0%) | -0.202 (-76.2%) |
| test_hard | SFT | BASE | +0.000 (n/a) | +0.000 (n/a) | +0.118 (n/a) | +0.202 (+319.6%) |
| test_hard | GRPO | BASE | +0.000 (n/a) | +0.000 (n/a) | +0.154 (n/a) | +0.226 (+357.2%) |
| test_hard | GRPO | SFT | +0.000 (n/a) | +0.000 (n/a) | +0.036 (+30.8%) | +0.024 (+9.0%) |

## Per-family breakdown (test)

| family | checkpoint | pass@1 | task completion | tool schema valid | perception recall |
|---|---|---:|---:|---:|---:|
| chart_agg | BASE | 0.000 | 0.000 | 0.000 | 0.100 |
| chart_agg | SFT | 0.000 | 0.000 | 0.000 | 0.050 |
| chart_agg | GRPO | 0.000 | 0.000 | 0.000 | 0.050 |
| doc_table | BASE | 0.000 | 0.000 | 0.000 | 0.094 |
| doc_table | SFT | 0.000 | 0.000 | 0.500 | 0.438 |
| doc_table | GRPO | 0.000 | 0.000 | 0.250 | 0.438 |
| screenshot_ui | BASE | 0.000 | 0.000 | 0.000 | 0.125 |
| screenshot_ui | SFT | 0.000 | 0.000 | 0.000 | 0.000 |
| screenshot_ui | GRPO | 0.000 | 0.000 | 0.000 | 0.000 |
| multi_frame | BASE | 0.000 | 0.000 | 0.000 | 0.083 |
| multi_frame | SFT | 0.000 | 0.000 | 0.750 | 0.500 |
| multi_frame | GRPO | 0.000 | 0.000 | 0.625 | 0.500 |

## Failure attribution (test)

| checkpoint | perception | planning | tool_schema | execution | arithmetic | format | budget | success | dominant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BASE | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | FORMAT |
| SFT | 3 | 1 | 0 | 0 | 0 | 7 | 5 | 0 | FORMAT |
| GRPO | 4 | 1 | 1 | 0 | 0 | 5 | 5 | 0 | FORMAT |


| checkpoint | perception | planning | tool_schema | execution | arithmetic | format | budget | success | dominant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BASE | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | FORMAT |
| SFT | 6 | 0 | 0 | 1 | 0 | 9 | 0 | 0 | FORMAT |
| GRPO | 5 | 0 | 1 | 1 | 0 | 9 | 0 | 0 | FORMAT |

## Training and cost

| stage | steps | wall clock | signal |
|---|---:|---:|---|
| SFT (LoRA) | 150 | 25.3 min | final val loss 0.1342 |
| GRPO | 3 | 7.5 min | reward 0.194 -> 0.018, completion 0.000 -> 0.000, skipped groups 1 |

GRPO throughput measured during this run: **1.6 rollouts/min**, mean step **112.5 s**, peak memory **5281 MB**, KL beta 0.02, clip eps 0.2.

## What improved, what did not

- **GRPO did not change pass@1** vs SFT on test.
- Multi-step task completion was unchanged vs SFT.
- Tool-call schema validity was unchanged vs SFT.
- Perception recall was unchanged vs SFT.
- ReAct format compliance changed by +0.125 vs SFT on test.
- Mean turns per episode changed by +0.31 (positive = shorter trajectories).
- Absolute GRPO test numbers: pass@1 **0.000**, completion **0.000**, schema **0.500**; on test_hard pass@1 **0.000**.
- Dominant failure bucket: **FORMAT** for BASE -> **FORMAT** for GRPO (test).
- Buckets reduced by GRPO: FORMAT (-11).

## Limitations

- **Small model / single hardware tier.** All numbers are from the `tiny` tier on `mps`; the `small`/`base` CUDA tiers are wired but not exercised on this machine.
- **Synthetic task suite.** Tasks are generated locally from seed 1234 and are not a substitute for real-world chart/document/screenshot benchmarks.
- **Small evaluation sample.** 4 items per split (stratified across 4 families) with k=4; pass@4 therefore rests on a handful of items per family and has wide error bars.
- **Single training seed.** SFT and GRPO were run once with seed 1234; no seed-variance error bars are reported.
- **Teacher-forced SFT targets** come from executing the gold tool sequence; the model never sees alternative valid strategies during cold start.
- **Perception recall** counts facts that appear anywhere in the model text, tool arguments or tool observations; a retrieval tool call can therefore surface a fact without the model reading it.

## Figures

![GRPO reward curve](figures/reward_curve.png)

![task completion vs GRPO step](figures/completion_vs_step.png)

![failure mix across checkpoints](figures/failure_mix.png)

![per-family pass@1](figures/per_family_pass1.png)

![reward component breakdown](figures/reward_components.png)

![BASE vs SFT vs GRPO summary](figures/summary_metrics.png)

<!-- RESULTS_END -->

Full tables, all splits, training details and the failure attribution are in
[`results/RESULTS.md`](results/RESULTS.md). Annotated trajectories are in
[`results/examples/`](results/examples/): a real planning failure and a real perception failure.
No successful held-out rollout was observed in the measured run, so no success example was fabricated.

---

## 5. Reproducibility

* **Task suite**: generated locally from `seed: 1234` (config) with pure `random.Random`/NumPy
  seeding; `tests/test_tasks.py` asserts identical item hashes and identical PNG bytes for the same
  seed. No dataset download is used for the core suite.
* **Seeds**: evaluation uses `eval.temperature: 0.8` and `k: 4`; the effective seed is recorded in
  the config block of `results/results.json`. SFT and GRPO use the config seed; GRPO additionally
  uses `grpo.seed`.
* **Commit**: the commit sha at evaluation time is stored in `environment.commit_sha`.
* **Exact commands** that produced `results/results.json`:

```bash
make data
make eval-base
make sft
make grpo
make eval-all     # writes results/results.json (reuses existing eval JSONs)
make report       # rewrites README + results/RESULTS.md from results.json
make bench        # writes results/benchmark.json + docs/scaling.md
```

* **Tests / CI**: `pytest -q` (unit + a full stub-tier pipeline test) and GitHub Actions on Python
  3.10/3.11 with `VLMGRPO_TIER=stub` and `HF_HUB_OFFLINE=1`.
* Artifacts ignored by git (regenerate with the commands above): `runs/`, `data/images/`,
  `results/trajectories/`.

---

## License

MIT - see [`LICENSE`](LICENSE). Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).
