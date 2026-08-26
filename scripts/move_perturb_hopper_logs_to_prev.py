#!/usr/bin/env python3
"""Move PerturbHopper-v4_* log directories under noise_assignment/prev.

The destination preserves each directory's path relative to noise_assignment,
so repeated names like PerturbHopper-v4_1 do not collide.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT = Path("logs/fed_ampo/tuned_mujoco/fixed/noise_assignment")
PATTERN = "PerturbHopper-v4_*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move PerturbHopper-v4_* logs under noise_assignment/prev."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory. Default: <root>/prev",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move directories. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing destination directory.",
    )
    return parser.parse_args()


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def find_hopper_logs(root: Path, dest: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(PATTERN)
        if path.is_dir() and not is_inside(path, dest)
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dest = (args.dest or root / "prev2").resolve()

    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    targets = find_hopper_logs(root, dest)
    if not targets:
        print(f"No {PATTERN} directories found under {root}")
        return

    moves: list[tuple[Path, Path]] = []
    for source in targets:
        target = dest / source.relative_to(root)
        moves.append((source, target))

    collisions = [target for _, target in moves if target.exists()]
    if collisions and not args.overwrite:
        print("Destination collision(s) found. Re-run with --overwrite if this is intentional:")
        for target in collisions[:20]:
            print(f"  {target}")
        if len(collisions) > 20:
            print(f"  ... and {len(collisions) - 20} more")
        raise SystemExit(1)

    action = "MOVE" if args.execute else "DRY-RUN"
    print(f"{action}: {len(moves)} directories")
    print(f"Root: {root}")
    print(f"Destination: {dest}")
    for source, target in moves:
        print(f"{source} -> {target}")

    if not args.execute:
        print("\nNo files moved. Re-run with --execute to apply.")
        return

    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and args.overwrite:
            shutil.rmtree(target)
        shutil.move(str(source), str(target))

    print(f"\nMoved {len(moves)} directories.")


if __name__ == "__main__":
    main()
