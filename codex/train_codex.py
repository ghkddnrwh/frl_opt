from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_ROOT = PROJECT_ROOT / "codex" / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

from codex.fedsp_pg_ppo_codex import CodexFedSPPGPPO
from rl_zoo3 import utils as zoo_utils


zoo_utils.ALGOS["fedsp_pg_ppo_codex"] = CodexFedSPPGPPO

from rl_zoo3 import train as zoo_train  # noqa: E402


zoo_train.ALGOS["fedsp_pg_ppo_codex"] = CodexFedSPPGPPO


if __name__ == "__main__":
    zoo_train.train()
