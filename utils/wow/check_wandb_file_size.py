import os
import wandb

ENTITY = "ukjo19"
PROJECT = "sb3"
TARGET = "output.log"

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}", per_page=100)

total_runs = 0
runs_with_files = 0
remaining_target_count = 0
visible_total_bytes = 0

print("Scanning...\n")

for run_idx, run in enumerate(runs, start=1):
    total_runs += 1

    try:
        files = list(run.files(per_page=100))
    except Exception as e:
        print(f"[FAILED] run={run.id} name={run.name}: {e!r}")
        continue

    total_size = 0
    targets = []

    for f in files:
        size = getattr(f, "size", 0) or 0
        total_size += size

        if os.path.basename(f.name) == TARGET:
            targets.append(f)

    visible_total_bytes += total_size

    if files:
        runs_with_files += 1

    if targets:
        remaining_target_count += len(targets)
        print(
            f"[TARGET STILL EXISTS] "
            f"run={run.id} name={run.name} "
            f"count={len(targets)}"
        )
        for f in targets:
            print(
                f"    {f.name} "
                f"({(getattr(f, 'size', 0) or 0) / 1024 / 1024:.2f} MB)"
            )

    # API에서 보이는 파일 총량이 큰 run만 출력
    if total_size > 1 * 1024 * 1024:
        print(
            f"[VISIBLE FILES] "
            f"run={run.id} name={run.name} "
            f"files={len(files)} "
            f"size={total_size / 1024 / 1024:.2f} MB"
        )

print("\n==============================")
print(f"total runs: {total_runs}")
print(f"runs with visible files: {runs_with_files}")
print(f"remaining output.log: {remaining_target_count}")
print(
    f"visible file total: "
    f"{visible_total_bytes / 1024 / 1024 / 1024:.3f} GB"
)