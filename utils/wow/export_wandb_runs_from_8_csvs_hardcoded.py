# export_wandb_runs_2026_08_29_evening_hardcoded.py
#
# Generated from 4 W&B CSV exports uploaded on 2026-08-29 evening.
# CSV files are NOT required when running this script; all run IDs are hardcoded.
#
# CSV rows / IDs including duplicates: 50
# Unique run IDs: 50
# Duplicate occurrences: 0
# CSV state counts: {'finished': 50}
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_08_29_evening_hardcoded.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_08_29_evening"
PAGE_SIZE = 500

# True: export only runs whose current W&B API state is "finished".
# False: export every hardcoded run using whatever history currently exists.
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'wandb_export_2026-08-29T20_42_30.557+09_00.csv': {'group': 'perturbhopper_v4_friction', 'env': 'PerturbHopper-v4', 'perturbation': 'friction', 'rows': 20, 'run_ids': 20, 'state_counts': {'finished': 20}}, 'wandb_export_2026-08-29T20_42_51.356+09_00.csv': {'group': 'perturbhopper_v4_gravity', 'env': 'PerturbHopper-v4', 'perturbation': 'gravity', 'rows': 20, 'run_ids': 20, 'state_counts': {'finished': 20}}, 'wandb_export_2026-08-29T20_43_07.719+09_00.csv': {'group': 'perturbhopper_v4_gravity', 'env': 'PerturbHopper-v4', 'perturbation': 'gravity', 'rows': 5, 'run_ids': 5, 'state_counts': {'finished': 5}}, 'wandb_export_2026-08-29T20_43_19.000+09_00.csv': {'group': 'perturbhopper_v4_friction', 'env': 'PerturbHopper-v4', 'perturbation': 'friction', 'rows': 5, 'run_ids': 5, 'state_counts': {'finished': 5}}}

RUN_IDS_BY_GROUP = {
    "perturbhopper_v4_friction": [
        "wykhp5mr",
        "8bidooiw",
        "kvgfuet6",
        "4s7gp10v",
        "srz1lpzy",
        "hjwxyzjn",
        "99lnoq39",
        "2pd6fyr6",
        "xh9uvvh3",
        "mrxye88o",
        "8zdy58mt",
        "pju4dhm5",
        "m6moi5jn",
        "wh3hxxhz",
        "0f8m67n8",
        "e59qp6y7",
        "i3np0e2w",
        "6ywiwr2z",
        "dtbsfuw2",
        "0m9ys7aw",
        "f74u7mf0",
        "y1jjftpi",
        "bhaifkqs",
        "1005exr7",
        "wna3xarb",
    ],
    "perturbhopper_v4_gravity": [
        "9piffegk",
        "ht44qec9",
        "1nekt7v8",
        "sp2y5lo3",
        "8t7xgj4b",
        "dey2ym7r",
        "tyomjoc7",
        "ms0f0yb1",
        "ekcrueff",
        "jnrxu6qq",
        "3vvak6zi",
        "6pa4a3cm",
        "fejdju9y",
        "sz9884sp",
        "iihki5z3",
        "575x6kvn",
        "5y5403pw",
        "v9izyhac",
        "a1km3pnl",
        "n64nrogv",
        "u1izjz8m",
        "xljees54",
        "7iugacj3",
        "pcojcy1h",
        "tnxrqg3k",
    ],
}

RUN_IDS = list(
    dict.fromkeys(
        run_id
        for ids in RUN_IDS_BY_GROUP.values()
        for run_id in ids
    )
)


def json_safe(x):
    if x is None:
        return None

    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x

    if isinstance(x, (str, int, bool)):
        return x

    if isinstance(x, datetime):
        return x.isoformat()

    try:
        import numpy as np

        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            x = float(x)
            return None if math.isnan(x) or math.isinf(x) else x

        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    return str(x)


def dump_json(path: Path, obj):
    path.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def group_for_run(run_id: str):
    for group, ids in RUN_IDS_BY_GROUP.items():
        if run_id in ids:
            return group
    return "unknown"


def export_run(run, out_root: Path):
    run_id = str(run.id)
    group = group_for_run(run_id)

    out = out_root / group / run_id
    out.mkdir(parents=True, exist_ok=True)

    config = {
        k: v
        for k, v in dict(run.config).items()
        if not str(k).startswith("_")
    }
    dump_json(out / "config.json", config)

    try:
        summary = dict(run.summary)
    except Exception:
        try:
            summary = run.summary._json_dict
        except Exception:
            summary = {}
    dump_json(out / "summary.json", summary)

    try:
        api_path = "/".join(run.path)
    except Exception:
        api_path = f"{ENTITY}/{PROJECT}/{run_id}"

    metadata = {
        "experiment_group": group,
        "api_path": api_path,
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "created_at": getattr(run, "created_at", None),
        "url": getattr(run, "url", None),
        "tags": getattr(run, "tags", None),
        "notes": getattr(run, "notes", None),
    }
    dump_json(out / "metadata.json", metadata)

    n_rows = 0
    all_keys = set()
    first_rows = []
    last_rows = []

    with (out / "history.jsonl").open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=PAGE_SIZE):
            row = json_safe(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_rows += 1
            all_keys.update(row.keys())

            if len(first_rows) < 5:
                first_rows.append(row)

            last_rows.append(row)
            if len(last_rows) > 5:
                last_rows.pop(0)

    readme = [
        "# W&B Run Export",
        "",
        f"- Experiment group: `{group}`",
        f"- API path: `{api_path}`",
        f"- Run name: `{getattr(run, 'name', None)}`",
        f"- Run id: `{run_id}`",
        f"- State: `{getattr(run, 'state', None)}`",
        f"- History rows: `{n_rows}`",
        "",
        "## Files",
        "- `metadata.json`",
        "- `config.json`",
        "- `summary.json`",
        "- `history.jsonl`",
        "",
        "## Logged keys",
        ", ".join(f"`{k}`" for k in sorted(all_keys)),
        "",
        "## Summary",
        "```json",
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Config",
        "```json",
        json.dumps(json_safe(config), ensure_ascii=False, indent=2),
        "```",
        "",
        "## First 5 history rows",
        "```json",
        json.dumps(first_rows, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Last 5 history rows",
        "```json",
        json.dumps(last_rows, ensure_ascii=False, indent=2),
        "```",
    ]
    (out / "README_for_GPT.md").write_text(
        "\n".join(readme),
        encoding="utf-8",
    )

    return {
        "id": run_id,
        "group": group,
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "history_rows": n_rows,
        "out_dir": str(out),
    }


def main():
    api = wandb.Api()
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] unique runs: {len(RUN_IDS)}")
    for group, ids in RUN_IDS_BY_GROUP.items():
        print(f"  - {group}: {len(ids)}")
    print()

    exported = []
    skipped = []
    errors = []

    for i, run_id in enumerate(RUN_IDS, start=1):
        group = group_for_run(run_id)
        api_path = f"{ENTITY}/{PROJECT}/{run_id}"

        print(
            f"[{i:03d}/{len(RUN_IDS):03d}] "
            f"{group} / {run_id}"
        )

        try:
            run = api.run(api_path)
            state = str(getattr(run, "state", "")).lower()

            if ONLY_FINISHED and state != "finished":
                print(f"    SKIP: state={state}")
                skipped.append({
                    "id": run_id,
                    "group": group,
                    "state": state,
                })
                continue

            result = export_run(run, out_root)
            exported.append(result)
            print(
                f"    DONE: {result['history_rows']} history rows"
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append({
                "id": run_id,
                "group": group,
                "api_path": api_path,
                "error": str(e),
            })

    dump_json(
        out_root / "_run_ids_by_group.json",
        RUN_IDS_BY_GROUP,
    )

    dump_json(
        out_root / "_source_file_metadata.json",
        SOURCE_FILE_METADATA,
    )

    dump_json(
        out_root / "_export_summary.json",
        {
            "entity": ENTITY,
            "project": PROJECT,
            "unique_run_count": len(RUN_IDS),
            "only_finished": ONLY_FINISHED,
            "expected_by_group": {
                g: len(ids)
                for g, ids in RUN_IDS_BY_GROUP.items()
            },
            "exported_count": len(exported),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "exported": exported,
            "skipped": skipped,
            "errors": errors,
        },
    )

    print()
    print("=" * 72)
    print(f"Expected : {len(RUN_IDS)}")
    print(f"Exported : {len(exported)}")
    print(f"Skipped  : {len(skipped)}")
    print(f"Errors   : {len(errors)}")
    print(f"Output   : {out_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
