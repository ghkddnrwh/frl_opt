from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "ablation_00_build_cache.py",
    "ablation_01_adaptive_vs_uniform.py",
    "ablation_02_dual_tracking_alignment.py",
    "ablation_03_lambda_dynamics.py",
    "ablation_04_worst_environment_switching.py",
    "ablation_05_dual_lr_timescale.py",
    "ablation_06_robustness_scalability_tradeoff.py",
    "ablation_07_focus_response.py",
    "ablation_08_gradient_diagnostics.py",
    "ablation_09_lambda_cap_activity.py",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Run all AMPO ablation analyses sequentially.")
    p.add_argument("--log-root", type=Path, default=Path("logs/wandb_logs_final_160"))
    p.add_argument("--out-root", type=Path, default=Path("logs/iclr2027_ampo/ablation"))
    p.add_argument("--late-evals", type=int, default=10)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        cmd = [
            sys.executable,
            str(here / name),
            "--log-root",
            str(args.log_root),
            "--out-root",
            str(args.out_root),
            "--late-evals",
            str(args.late_evals),
        ]
        print("\n" + "=" * 88)
        print("[Run]", " ".join(cmd))
        print("=" * 88)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
