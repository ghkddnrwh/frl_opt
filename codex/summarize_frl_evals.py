from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def summarize(path: Path) -> tuple[float, float, float, float, int, int]:
    data = np.load(path, allow_pickle=True)
    local_min = np.asarray(data["local_min_across_clients"], dtype=float)
    local_mean = np.asarray(data["local_mean_across_clients"], dtype=float)
    nominal_mean = np.asarray(data["nominal_mean_across_clients"], dtype=float)
    rounds = np.asarray(data["rounds"], dtype=int)
    local_steps = int(data["local_steps"]) if "local_steps" in data.files else -1
    return (
        float(local_min[-1]),
        float(np.nanmax(local_min)),
        float(local_mean[-1]),
        float(nominal_mean[-1]),
        int(rounds[-1]) if rounds.size else 0,
        local_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        for path in sorted(Path(root).glob("**/evaluations.npz")):
            try:
                final_min, best_min, final_mean, final_nominal, rounds, local_steps = summarize(path)
            except Exception as exc:  # pragma: no cover - diagnostic script
                print(f"skip {path}: {exc}")
                continue
            rows.append((final_min, best_min, final_mean, final_nominal, rounds, local_steps, path))

    for final_min, best_min, final_mean, final_nominal, rounds, local_steps, path in sorted(rows, reverse=True):
        print(
            f"{final_min:9.2f} best={best_min:9.2f} "
            f"mean={final_mean:9.2f} nominal={final_nominal:9.2f} "
            f"rounds={rounds:5d} local_steps={local_steps:5d} {path}"
        )


if __name__ == "__main__":
    main()

