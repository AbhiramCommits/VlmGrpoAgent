# Contributing

Thanks for looking at VLM-GRPO. The project is intentionally small and fully
offline-testable.

## Quickstart

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

make smoke        # offline stub rollout, no weights download
make data         # deterministic suite + PNGs
make test         # unit + stub-tier end-to-end tests
make lint         # ruff
```

Run the real tiny-tier pipeline (Apple MPS / CPU / CUDA):

```bash
make eval-base
make sft
make grpo
make eval-all
make report
```

## Ground rules

* **No invented numbers.** Every metric in `README.md` / `results/RESULTS.md`
  comes from `results/results.json`, which is written by `make eval-all`.
  Change a number only by re-running the pipeline and `make report`.
* **Determinism.** Task generation, SFT data and evaluation must be reproducible
  from the configured seed. Never add unseeded randomness to `vlmgrpo/tasks/`.
* **No network in the core suite.** Task images are rendered locally; the CI
  workflow runs with `HF_HUB_OFFLINE=1` on the `stub` tier.
* **Sandbox changes need tests.** Any change to `vlmgrpo/tools.py` must keep the
  import allowlist, restricted builtins, audit hook, cwd isolation and timeout,
  and must add a positive/negative case to `tests/test_tools.py`.

## Pull requests

1. `ruff check .`
2. `pytest -q` (the stub end-to-end test should stay under ~3 minutes)
3. If you changed evaluation or analysis, run `make eval-all && make report`
   and commit the regenerated `results/results.json`, `results/RESULTS.md` and
   figures together with the README.

## Layout

See the file map in `README.md`. Training entrypoints (`train_sft.py`,
`train_grpo.py`, `train_grpo_trl.py`) are excluded from the coverage gate;
everything else is expected to be covered by tests.
