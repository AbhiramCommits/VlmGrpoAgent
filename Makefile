# ---------------------------------------------------------------------------
# VLM-GRPO Makefile.  CONFIG selects the experiment; PY prefers the local venv.
# ---------------------------------------------------------------------------
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3.11 >/dev/null 2>&1; then echo python3.11; else echo python3; fi)
CONFIG ?= configs/tiny.yaml
PYTEST ?= $(PY) -m pytest -q
RUFF ?= $(PY) -m ruff

.PHONY: setup smoke data eval-base sft grpo eval-sft eval-grpo eval-all test lint report bench clean help

help:
	@echo "targets: setup smoke data eval-base sft grpo eval-all test lint report bench"

setup:                ## create .venv and install pinned deps
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

smoke:                ## offline stub rollouts on one rendered task
	$(PY) scripts/smoke.py --config $(CONFIG)

data:                 ## generate the deterministic task suite + PNGs + stats
	$(PY) -m vlmgrpo.tasks.generate --config $(CONFIG)

eval-base:            ## BASE model zero-shot on val/test/test_hard
	$(PY) -m vlmgrpo.evaluate --config $(CONFIG) --checkpoint base --splits val,test,test_hard

sft:                  ## build gold SFT data + LoRA SFT cold start
	$(PY) -m vlmgrpo.data.build_sft --config $(CONFIG)
	$(PY) -m vlmgrpo.train_sft --config $(CONFIG)

grpo:                 ## GRPO/RLVR online RL from the SFT checkpoint
	$(PY) -m vlmgrpo.train_grpo --config $(CONFIG)

eval-sft:             ## SFT checkpoint on val/test/test_hard
	$(PY) -m vlmgrpo.evaluate --config $(CONFIG) --checkpoint sft --splits val,test,test_hard

eval-grpo:            ## GRPO checkpoint on val/test/test_hard
	$(PY) -m vlmgrpo.evaluate --config $(CONFIG) --checkpoint grpo --splits val,test,test_hard

eval-all:             ## full {BASE,SFT,GRPO} x {test,test_hard} matrix
	$(PY) -m vlmgrpo.analysis.compare --config $(CONFIG)

test:                 ## unit + end-to-end tests
	$(PYTEST)

lint:                 ## ruff
	$(RUFF) check .

report:               ## regenerate results/RESULTS.md, README tables and figures
	$(PY) scripts/make_results_table.py --config $(CONFIG)

bench:                ## measure rollout throughput for docs/scaling.md
	$(PY) scripts/benchmark_rollouts.py --config $(CONFIG)

clean:
	rm -rf runs results/trajectories .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
