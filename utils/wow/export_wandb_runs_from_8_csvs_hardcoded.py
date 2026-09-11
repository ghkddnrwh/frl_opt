# export_wandb_runs_2026_09_11_hardcoded_old_style.py
#
# 2026-09-11 W&B CSV의 run ID를 직접 하드코딩한 exporter.
# 실행 시 CSV 파일은 필요하지 않습니다.
#
# CSV rows: 85
# Unique run IDs: 85
# State counts: {'finished': 70, 'running': 10, 'crashed': 5}
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_09_11_hardcoded_old_style.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_09_11"
PAGE_SIZE = 500

# True면 API 조회 시점에 finished인 run만 export
# False면 CSV에 포함된 모든 run의 현재까지 로그를 export
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'wandb_export_2026-09-11T13_43_38.917+09_00.csv': {'rows': 85,
                                                    'run_ids': 85,
                                                    'unique_run_ids': 85,
                                                    'state_counts': {'finished': 70, 'running': 10, 'crashed': 5},
                                                    'group_counts': {'ant_friction': 35, 'ant_gravity': 50}}}


RUN_IDS_BY_GROUP = {'ant_friction': ['f28zs4lv',
                  '23y13v46',
                  'h8npgcqy',
                  '402uf31e',
                  'etlpas81',
                  'ou0rh6tf',
                  'djtninpi',
                  'dszyhz2o',
                  'dvsmrw60',
                  'bstzx2nt',
                  'bli50l87',
                  'rq3l2kgw',
                  '9qxlri3n',
                  'tkrrfavo',
                  '91y3ygr2',
                  '5h4d24aa',
                  'u00taxv3',
                  '6xsvuja5',
                  '1i6cnt64',
                  'wqmggqvg',
                  'ja3c3as0',
                  'pygy749k',
                  'dock5lrf',
                  'hz9bsqmd',
                  'x88852kl',
                  '1ja51ty9',
                  'wu1llop0',
                  '8sjnxhao',
                  'rdlwbqdb',
                  'qompjwdz',
                  'jwf8snum',
                  '12vx94ul',
                  '2885masn',
                  'lw1n6s5o',
                  '2qck97c6'],
 'ant_gravity': ['8pf5kn0n',
                 'sz0ev9xg',
                 '0i9s22kc',
                 'xa3jhcry',
                 '4uqcxc7q',
                 'px8klaqv',
                 'hrtbq363',
                 '8d0ohzt6',
                 'on4eilez',
                 'snafa5v1',
                 'lu58dwm3',
                 'hist6g6q',
                 'xnvecfjq',
                 '440zo275',
                 'hpaufgz6',
                 '0w8x8po5',
                 '2jk92wj0',
                 'mg720wy1',
                 '1ds7ri80',
                 'm1v6m64t',
                 'fhw3xa97',
                 'pm0z3znj',
                 '21slmgxf',
                 '22tt4g07',
                 'u37wtyta',
                 'bafbxve8',
                 'uw0q3a6v',
                 'wlsjahsq',
                 '6qra1bqf',
                 'ufdow4cb',
                 'kuqgkcal',
                 '2dki2zwo',
                 'it7pu58h',
                 'y55viycr',
                 'lv53hx1r',
                 'jr20o7yh',
                 '3n78no1b',
                 'izccf1lc',
                 'oc8d9bhl',
                 'txd9r951',
                 'qe41rubq',
                 'd3f5rvdk',
                 'k7cxep04',
                 '6avkzu2l',
                 'kcosxef0',
                 'xqimgnli',
                 'glkpb52d',
                 '6yin9imk',
                 'm4zwwwhb',
                 'wokyijat']}


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
        "entity": getattr(
            run,
            "entity",
            None,
        ),
        "project": getattr(
            run,
            "project",
            None,
        ),
        "id": getattr(
            run,
            "id",
            None,
        ),
        "name": getattr(
            run,
            "name",
            None,
        ),
        "state": getattr(
            run,
            "state",
            None,
        ),
        "created_at": getattr(
            run,
            "created_at",
            None,
        ),
        "url": getattr(
            run,
            "url",
            None,
        ),
        "tags": getattr(
            run,
            "tags",
            None,
        ),
        "notes": getattr(
            run,
            "notes",
            None,
        ),
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

            all_keys.update(
                row.keys()
            )

            if len(first_rows) < 5:
                first_rows.append(
                    row
                )

            last_rows.append(
                row
            )

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
        "name": getattr(
            run,
            "name",
            None,
        ),
        "state": getattr(
            run,
            "state",
            None,
        ),
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
        f"[INFO] unique runs: "
        f"{len(RUN_IDS)}"
    )

    for group, ids in (
        RUN_IDS_BY_GROUP.items()
    ):
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
                    f"    SKIP: "
                    f"state={state}"
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
            "unique_run_count": len(
                RUN_IDS
            ),
            "only_finished": (
                ONLY_FINISHED
            ),
            "expected_by_group": {
                group: len(ids)
                for group, ids
                in RUN_IDS_BY_GROUP.items()
            },
            "exported_count": len(
                exported
            ),
            "skipped_count": len(
                skipped
            ),
            "error_count": len(
                errors
            ),
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
