# export_wandb_runs_2026_08_24_complete_4groups.py
#
# 4 experiment groups, 100 unique W&B runs:
#   Ant + friction      : 35
#   Ant + gravity       : 35
#   Walker2d + gravity  : 15
#   Walker2d + friction : 15
#
# The duplicate Ant+gravity CSV is intentionally excluded.
# CSV files are not needed at runtime; run IDs are hardcoded below.
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_08_24_complete_4groups.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb

ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_08_24_complete_4groups"
PAGE_SIZE = 500
ONLY_FINISHED = True

GROUP_METADATA = {
    "ant_friction": {"source_csv": "wandb_export_2026-08-24T00_06_01.484+09_00.csv", "env": ['PerturbAnt-v4'], "perturbation": ['friction'], "count": 35, "states": {'finished': 35}},
    "ant_gravity": {"source_csv": "wandb_export_2026-08-24T00_06_20.716+09_00.csv", "env": ['PerturbAnt-v4'], "perturbation": ['gravity'], "count": 35, "states": {'finished': 35}},
    "walker2d_gravity": {"source_csv": "wandb_export_2026-08-24T00_06_33.215+09_00.csv", "env": ['PerturbWalker2d-v4'], "perturbation": ['gravity'], "count": 15, "states": {'finished': 15}},
    "walker2d_friction": {"source_csv": "wandb_export_2026-08-24T00_12_55.021+09_00.csv", "env": ['PerturbWalker2d-v4'], "perturbation": ['friction'], "count": 15, "states": {'finished': 15}},
}

RUN_IDS_BY_GROUP = {
    "ant_friction": [
        "m49c7ag2",
        "8xvs0mrg",
        "4sknnqj9",
        "0eq45zpc",
        "0x9wuc1u",
        "m86azz8r",
        "teem7l7c",
        "659mx9vi",
        "ct1lkfu1",
        "uvfhldbg",
        "5sk8zlks",
        "u67n9c88",
        "363b0zgv",
        "kvk6xhfb",
        "eatii1by",
        "amyo6peh",
        "d4d03150",
        "mjfth54l",
        "raqdcayd",
        "4ew5fejc",
        "62dlvguj",
        "o5resmb6",
        "js1nth7z",
        "ekyrewb2",
        "xgo2tyfm",
        "wej3irel",
        "32zny8nn",
        "053ckiqq",
        "dsnspggw",
        "eroqqast",
        "9n5x8ih7",
        "7lr0est2",
        "08q204s5",
        "10pl01ao",
        "ijyohmzp",
    ],
    "ant_gravity": [
        "j090ufki",
        "8k0qrrj5",
        "c5rjjbpx",
        "7l7956xn",
        "cuywj0js",
        "dpq213x7",
        "vagr0o1q",
        "z6jz5map",
        "at0jk2ve",
        "53kkgurk",
        "jwvvjhal",
        "n3dxns16",
        "pkav0p5o",
        "ylvda79k",
        "u6hge6bw",
        "3b4hcbzu",
        "mcfxlxbm",
        "mtvcsajj",
        "87xavuk5",
        "weew8pn6",
        "b9izgrur",
        "8o6ngzi9",
        "uvaqznbu",
        "rzvwnl0d",
        "cj2iqmwb",
        "1y2frlb7",
        "h2mhzqrc",
        "3oxacb9p",
        "9snsffek",
        "w2u2p5xb",
        "pbhakfn0",
        "r5w02zk7",
        "fsauv9z9",
        "adwfq6jf",
        "cih38gzn",
    ],
    "walker2d_gravity": [
        "oyn511sj",
        "d8684h9e",
        "m68ruxw6",
        "kn5mroxl",
        "z459p4f8",
        "we3b9ti4",
        "u015w1uu",
        "7xxbfoa1",
        "6suttm0h",
        "uwdk7j81",
        "52vf1jze",
        "26xl01hp",
        "dbv34ot3",
        "lwgppgub",
        "tfmseudm",
    ],
    "walker2d_friction": [
        "0z30cepz",
        "axrfb5ec",
        "o8vvco4o",
        "t3hik7ut",
        "50j0ubb3",
        "4bb24zsz",
        "k93cmvqg",
        "f6jyr0tu",
        "rc860tns",
        "rul0agim",
        "x5l4zhc1",
        "b8bqw6im",
        "zp3qw0bz",
        "ru1z3kx3",
        "wrbsmjhl",
    ],
}

RUN_IDS = list(dict.fromkeys(
    run_id
    for ids in RUN_IDS_BY_GROUP.values()
    for run_id in ids
))

assert len(RUN_IDS) == 100, f"Expected 100 unique run IDs, got {len(RUN_IDS)}"


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
        "group_metadata": GROUP_METADATA.get(group),
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
    (out / "README_for_GPT.md").write_text("\n".join(readme), encoding="utf-8")

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

    print("[INFO] 4 groups / 100 unique runs")
    for group, ids in RUN_IDS_BY_GROUP.items():
        print(f"  - {group}: {len(ids)}")
    print()

    exported = []
    skipped = []
    errors = []

    for i, run_id in enumerate(RUN_IDS, start=1):
        group = group_for_run(run_id)
        api_path = f"{ENTITY}/{PROJECT}/{run_id}"
        print(f"[{i:03d}/100] {group} / {run_id}")

        try:
            run = api.run(api_path)
            state = str(getattr(run, "state", "")).lower()

            if ONLY_FINISHED and state != "finished":
                print(f"    SKIP: state={state}")
                skipped.append({"id": run_id, "group": group, "state": state})
                continue

            result = export_run(run, out_root)
            exported.append(result)
            print(f"    DONE: {result['history_rows']} history rows")

        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append({
                "id": run_id,
                "group": group,
                "api_path": api_path,
                "error": str(e),
            })

    expected_by_group = {g: len(ids) for g, ids in RUN_IDS_BY_GROUP.items()}
    exported_by_group = {
        g: sum(1 for x in exported if x["group"] == g)
        for g in RUN_IDS_BY_GROUP
    }

    dump_json(out_root / "_run_ids_by_group.json", RUN_IDS_BY_GROUP)
    dump_json(out_root / "_export_summary.json", {
        "entity": ENTITY,
        "project": PROJECT,
        "expected_total": 100,
        "expected_by_group": expected_by_group,
        "exported_total": len(exported),
        "exported_by_group": exported_by_group,
        "skipped_total": len(skipped),
        "error_total": len(errors),
        "skipped": skipped,
        "errors": errors,
    })

    print()
    print("=" * 72)
    print(f"Expected : 100")
    print(f"Exported : {len(exported)}")
    print(f"Skipped  : {len(skipped)}")
    print(f"Errors   : {len(errors)}")
    print(f"Output   : {out_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
