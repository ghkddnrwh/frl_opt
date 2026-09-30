import os
import time
import wandb

ENTITY = "ukjo19"
PROJECT = "sb3"

DRY_RUN = False   # 먼저 True로 확인 → 문제 없으면 False

api = wandb.Api()

matched_count = 0
deleted_count = 0
matched_bytes = 0

failed_runs = []
failed_files = []


def safe_get_files(run):
    try:
        return list(run.files(per_page=100))

    except Exception as e:
        print(f"[RUN FILE LIST FAILED] run={run.id} name={run.name}")
        print(f"  error: {repr(e)}")

        failed_runs.append(
            (run.id, run.name, repr(e))
        )

        return []


runs = api.runs(
    f"{ENTITY}/{PROJECT}",
    per_page=100,
)

for run_idx, run in enumerate(runs, start=1):

    print(
        f"\n[RUN {run_idx}] "
        f"id={run.id} "
        f"name={run.name}"
    )

    files = safe_get_files(run)

    for f in files:

        basename = os.path.basename(f.name)

        # TensorBoard event 파일만 선택
        if not basename.startswith("events.out.tfevents."):
            continue

        size = getattr(f, "size", 0) or 0

        matched_count += 1
        matched_bytes += size

        print(
            f"[FOUND] "
            f"run={run.id} "
            f"size={size / 1024**2:.2f} MB\n"
            f"        {f.name}"
        )

        if not DRY_RUN:
            try:
                f.delete()

                deleted_count += 1

                print(f"[DELETED] {f.name}")

                # API에 너무 빠르게 요청하지 않도록
                time.sleep(0.2)

            except Exception as e:

                print(
                    f"[DELETE FAILED] "
                    f"run={run.id} "
                    f"file={f.name}"
                )

                print(f"  error: {repr(e)}")

                failed_files.append(
                    (run.id, f.name, repr(e))
                )


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"matched files : {matched_count}")
print(f"matched size  : {matched_bytes / 1024**2:.2f} MB")
print(f"matched size  : {matched_bytes / 1024**3:.2f} GB")
print(f"deleted files : {deleted_count}")
print(f"failed runs   : {len(failed_runs)}")
print(f"failed files  : {len(failed_files)}")


if failed_runs:
    print("\n===== FAILED RUNS =====")

    for run_id, run_name, err in failed_runs:
        print(
            f"run={run_id} "
            f"name={run_name} "
            f"error={err}"
        )


if failed_files:
    print("\n===== FAILED FILES =====")

    for run_id, file_name, err in failed_files:
        print(
            f"run={run_id} "
            f"file={file_name} "
            f"error={err}"
        )