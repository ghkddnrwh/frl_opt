# export_wandb_runs_2026_09_30_80_hardcoded_old_style.py
#
# 2026-09-30에 제공된 2개 W&B CSV의 run ID를 직접 하드코딩한 exporter.
# 실행 시 CSV 파일은 필요하지 않습니다.
#
# 구성:
#   PPO Avg       : 40 runs
#   AMPO Uniform  : 40 runs
#
# 각 알고리즘:
#   4 environments x 2 perturbations x 5 seeds = 40 runs
#
# Environments:
#   Ant
#   HalfCheetah
#   Hopper
#   Walker2d
#
# Perturbations:
#   friction
#   gravity
#
# Seeds:
#   1, 2, 3, 4, 5
#
# Total CSV rows: 80
# Unique run IDs: 80
# State counts: finished 80
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_09_30_80_hardcoded_old_style.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_09_30_80"
PAGE_SIZE = 500

# True면 API 조회 시점에 finished인 run만 export
# False면 하드코딩된 모든 run의 현재까지 로그를 export
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'wandb_export_2026-09-30T10_42_06.245+09_00.csv': {'algorithm': 'ppo_avg',
                                                    'rows': 40,
                                                    'run_ids': 40,
                                                    'unique_run_ids': 40,
                                                    'state_counts': {'finished': 40},
                                                    'envs': ['PerturbAnt-v4',
                                                             'PerturbHalfCheetah-v4',
                                                             'PerturbHopper-v4',
                                                             'PerturbWalker2d-v4'],
                                                    'perturbations': ['friction', 'gravity'],
                                                    'seeds': [1, 2, 3, 4, 5]},
 'wandb_export_2026-09-30T10_42_26.277+09_00.csv': {'algorithm': 'ampo_uniform',
                                                    'rows': 40,
                                                    'run_ids': 40,
                                                    'unique_run_ids': 40,
                                                    'state_counts': {'finished': 40},
                                                    'envs': ['PerturbAnt-v4',
                                                             'PerturbHalfCheetah-v4',
                                                             'PerturbHopper-v4',
                                                             'PerturbWalker2d-v4'],
                                                    'perturbations': ['friction', 'gravity'],
                                                    'seeds': [1, 2, 3, 4, 5]}}


RUN_IDS_BY_GROUP = {'ppo_avg_ant_friction': ['2lhtmky5', 'rsrfv7j9', '9tk5k6b1', 'uxm5v0ce', 'izsdo2kq'],
 'ppo_avg_ant_gravity': ['l6xh08px', '8cp86nve', 'dklcpqmw', '3r8n1vw1', 'ym2ka1a8'],
 'ppo_avg_halfcheetah_friction': ['romrukff', 'ojvm6ghr', 'z6ouxp18', 'xbpccmmm', 'f4wope86'],
 'ppo_avg_halfcheetah_gravity': ['de5t6qpx', 'wnwiwctp', 'rstfh92g', 'csm42l4w', 'r953j4hj'],
 'ppo_avg_hopper_friction': ['wmyh9icx', 'ue3lzxlr', 'dc9c5ml8', 'hwnf1iag', 'uz6vbv19'],
 'ppo_avg_hopper_gravity': ['9yl4586d', 'm4ihon4k', 'dk4igffu', 'fpej6xep', 'iq0pof9c'],
 'ppo_avg_walker2d_friction': ['e80ul4fu', 'zj60trz9', 'rr5o5p72', 'qc44wgcu', 'ne2z2ewe'],
 'ppo_avg_walker2d_gravity': ['s043chqn', 'lc8qh7mx', 'tydtiexw', '9iqf7d8s', 'dl0l7e70'],
 'ampo_uniform_ant_friction': ['3q8ikuas', 'w9rvjtpz', 'v1pj1ijc', 'gpki0p99', 'zulb79wp'],
 'ampo_uniform_ant_gravity': ['h2qp0w51', 'f7y2eg2p', 'o1y1275o', 'qwk5km9t', 's93twjbh'],
 'ampo_uniform_halfcheetah_friction': ['fnjvgeku', '7nada33z', 'jmp10v33', 'e1zwv7ch', '3rk57z7s'],
 'ampo_uniform_halfcheetah_gravity': ['y06pq5xb', 'muqfi2ag', '8uvzpd75', 'l2f7xacr', 'df8vm5h9'],
 'ampo_uniform_hopper_friction': ['m584m67w', '4b27wsu0', 'bqn41uz5', 'ngnywo3o', 'lzavbamd'],
 'ampo_uniform_hopper_gravity': ['gbgpdvdf', '0d7cp0yd', 'bmo35z2q', '8jx5z67o', '4tsg2syr'],
 'ampo_uniform_walker2d_friction': ['wpgyjfsx', 'oap0z0l7', '259vewuf', 'a820twec', 'lxmd3rc4'],
 'ampo_uniform_walker2d_gravity': ['89mptaq1', 'bytdcy07', 'dk4l8bxn', 'k4gucsh3', 'r72o6w4o']}


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
        return {
            str(k): json_safe(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            json_safe(v)
            for v in x
        ]

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

    out = (
        out_root
        / group
        / run_id
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. config
    config = {
        k: v
        for k, v in dict(run.config).items()
        if not str(k).startswith("_")
    }

    dump_json(
        out / "config.json",
        config,
    )

    # 2. summary
    try:
        summary = dict(run.summary)

    except Exception:
        try:
            summary = run.summary._json_dict

        except Exception:
            summary = {}

    dump_json(
        out / "summary.json",
        summary,
    )

    # 3. metadata
    try:
        api_path = "/".join(run.path)

    except Exception:
        api_path = (
            f"{ENTITY}/"
            f"{PROJECT}/"
            f"{run_id}"
        )

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

    dump_json(
        out / "metadata.json",
        metadata,
    )

    # 4. full scalar history
    n_rows = 0
    all_keys = set()
    first_rows = []
    last_rows = []

    history_path = (
        out
        / "history.jsonl"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in run.scan_history(
            page_size=PAGE_SIZE
        ):
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
        f"- Project: `{PROJECT}`",
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

    (
        out
        / "README_for_GPT.md"
    ).write_text(
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

    out_root = Path(
        OUT_ROOT
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[INFO] unique runs: {len(RUN_IDS)}"
    )

    for group, ids in RUN_IDS_BY_GROUP.items():
        print(
            f"  - {group}: "
            f"{len(ids)}"
        )

    print()

    exported = []
    skipped = []
    errors = []

    for i, run_id in enumerate(
        RUN_IDS,
        start=1,
    ):
        group = group_for_run(
            run_id
        )

        api_path = (
            f"{ENTITY}/"
            f"{PROJECT}/"
            f"{run_id}"
        )

        print(
            f"[{i:03d}/"
            f"{len(RUN_IDS):03d}] "
            f"{group} / "
            f"{run_id}"
        )

        try:
            run = api.run(
                api_path
            )

            state = str(
                getattr(
                    run,
                    "state",
                    "",
                )
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

            exported.append(
                result
            )

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
        out_root
        / "_run_ids_by_group.json",
        RUN_IDS_BY_GROUP,
    )

    dump_json(
        out_root
        / "_source_file_metadata.json",
        SOURCE_FILE_METADATA,
    )

    dump_json(
        out_root
        / "_export_summary.json",
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
        f"Expected : "
        f"{len(RUN_IDS)}"
    )

    print(
        f"Exported : "
        f"{len(exported)}"
    )

    print(
        f"Skipped  : "
        f"{len(skipped)}"
    )

    print(
        f"Errors   : "
        f"{len(errors)}"
    )

    print(
        f"Output   : "
        f"{out_root}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
