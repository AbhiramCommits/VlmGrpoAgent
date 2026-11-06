"""VLM-GRPO: verifiable-reward RL (GRPO) for multimodal tool-using agents."""

import os

# Tokenizers warns/fork-deadlocks when the tool sandbox forks after BPE use.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"

__all__ = ["__version__"]
