# export_wandb_runs_2026_09_10_hardcoded_old_style.py
#
# 2026-09-10 W&B CSV의 run ID를 직접 하드코딩한 exporter.
# 실행 시 CSV 파일은 필요하지 않습니다.
#
# CSV rows: 26
# Unique run IDs: 26
# State counts: {'finished': 20, 'running': 6}
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_09_10_hardcoded_old_style.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_09_10"
PAGE_SIZE = 500

# True면 API 조회 시점에 finished인 run만 export
# False면 CSV에 포함된 모든 run의 현재까지 로그를 export
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'wandb_export_2026-09-10T19_56_26.045+09_00.csv': {'rows': 26,
                                                    'run_ids': 26,
                                                    'state_counts': {'finished': 20, 'running': 6},
                                                    'group_counts': {'ant_gravity': 11, 'ant_friction': 15}}}


RUN_IDS_BY_GROUP = {'ant_gravity': ['txd9r951',
                 'qe41rubq',
                 'd3f5rvdk',
                 'k7cxep04',
                 '6avkzu2l',
                 'kcosxef0',
                 'xqimgnli',
                 'glkpb52d',
                 '6yin9imk',
                 'm4zwwwhb',
                 'wokyijat'],
 'ant_friction': ['jwf8snum',
                  '12vx94ul',
                  '2885masn',
                  'lw1n6s5o',
                  '2qck97c6',
                  'yfmjomu9',
                  '9onjj7of',
                  '8yghtffy',
                  'm57sott7',
                  'dr3pxzox',
                  'v6dr4xhz',
                  'kraa3swi',
                  '29gnja2l',
                  't7v7k2fh',
                  'c8f46z8i']}


RUN_IDS = list(dict.fromkeys(
    run_id
    for ids in RUN_IDS_BY_GROUP.values()
    for run_id in ids
))


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
        json.dumps(
            json_safe(obj),
            ensure_ascii=False,
            indent=2,
        ),
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

    # 1. config
    config = {
        k: v
        for k, v in dict(run.config).items()
        if not str(k).startswith("_")
    }
    dump_json(out / "config.json", config)

    # 2. summary
    try:
        summary = dict(run.summary)

    except Exception:
        try:
            summary = run.summary._json_dict

        except Exception:
            summary = {}

    dump_json(out / "summary.json", summary)

    # 3. metadata
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

    # 4. full scalar history
    n_rows = 0
    all_keys = set()
    first_rows = []
    last_rows = []

    history_path = out / "history.jsonl"

    with history_path.open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=PAGE_SIZE):
            row = json_safe(row)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            n_rows += 1
            all_keys.update(row.keys())

            if len(first_rows) < 5:
                first_rows.append(row)

            last_rows.append(row)

            if len(last_rows) > 5:
                last_rows.pop(0)

    # 5. README
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
        ", ".join(
            f"`{k}`"
            for k in sorted(all_keys)
        ),
        "",
        "## Summary",
        "```json",
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Config",
        "```json",
        json.dumps(
            json_safe(config),
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## First 5 history rows",
        "```json",
        json.dumps(
            first_rows,
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Last 5 history rows",
        "```json",
        json.dumps(
            last_rows,
            ensure_ascii=False,
            indent=2,
        ),
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
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[INFO] unique runs: {len(RUN_IDS)}"
    )

    for group, ids in RUN_IDS_BY_GROUP.items():
        print(
            f"  - {group}: {len(ids)}"
        )

    print()

    exported = []
    skipped = []
    errors = []

    for i, run_id in enumerate(
        RUN_IDS,
        start=1,
    ):
        group = group_for_run(run_id)
        api_path = (
            f"{ENTITY}/{PROJECT}/{run_id}"
        )

        print(
            f"[{i:03d}/{len(RUN_IDS):03d}] "
            f"{group} / {run_id}"
        )

        try:
            run = api.run(api_path)

            state = str(
                getattr(run, "state", "")
            ).lower()

            if (
                ONLY_FINISHED
                and state != "finished"
            ):
                print(
                    f"    SKIP: state={state}"
                )

                skipped.append({
                    "id": run_id,
                    "group": group,
                    "state": state,
                })

                continue

            result = export_run(
                run,
                out_root,
            )

            exported.append(result)

            print(
                f"    DONE: "
                f"{result['history_rows']} "
                f"history rows"
            )

        except Exception as e:
            print(
                f"    ERROR: {e}"
            )

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
                group: len(ids)
                for group, ids
                in RUN_IDS_BY_GROUP.items()
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
    print(
        f"Expected : {len(RUN_IDS)}"
    )
    print(
        f"Exported : {len(exported)}"
    )
    print(
        f"Skipped  : {len(skipped)}"
    )
    print(
        f"Errors   : {len(errors)}"
    )
    print(
        f"Output   : {out_root}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
