# # export_wandb_runs_2026_08_31_hardcoded.py
# # Generated from 9 W&B CSV exports.
# # CSV files are not needed at runtime; all unique run IDs are hardcoded.
# # Raw CSV rows: 161
# # Unique runs: 111
# # Duplicate occurrences removed: 50
# # CSV state counts before deduplication: {'finished': 105, 'running': 56}

# import json
# import math
# from datetime import datetime
# from pathlib import Path
# import wandb

# ENTITY = "ukjo19"
# PROJECT = "sb3"
# OUT_ROOT = "logs/wandb_logs_2026_08_31"
# PAGE_SIZE = 500

# # False: export every hardcoded run, including currently running ones.
# # True: export only runs whose current W&B API state is finished.
# ONLY_FINISHED = False

# SOURCE_FILE_METADATA = {'wandb_export_2026-08-31T18_10_23.444+09_00.csv': {'group': 'hopper_friction', 'env': 'PerturbHopper-v4', 'perturbation': 'friction', 'rows': 10, 'state_counts': {'finished': 10}}, 'wandb_export_2026-08-31T18_10_41.973+09_00.csv': {'group': 'hopper_gravity', 'env': 'PerturbHopper-v4', 'perturbation': 'gravity', 'rows': 6, 'state_counts': {'finished': 5, 'running': 1}}, 'wandb_export_2026-08-31T18_10_56.441+09_00.csv': {'group': 'ant_gravity', 'env': 'PerturbAnt-v4', 'perturbation': 'gravity', 'rows': 15, 'state_counts': {'finished': 15}}, 'wandb_export_2026-08-31T18_15_33.077+09_00.csv': {'group': 'ant_gravity', 'env': 'PerturbAnt-v4', 'perturbation': 'gravity', 'rows': 15, 'state_counts': {'finished': 15}}, 'wandb_export_2026-08-31T18_15_43.721+09_00.csv': {'group': 'ant_friction', 'env': 'PerturbAnt-v4', 'perturbation': 'friction', 'rows': 15, 'state_counts': {'finished': 15}}, 'wandb_export_2026-08-31T18_15_55.481+09_00.csv': {'group': 'halfcheetah_friction', 'env': 'PerturbHalfCheetah-v4', 'perturbation': 'friction', 'rows': 35, 'state_counts': {'running': 20, 'finished': 15}}, 'wandb_export_2026-08-31T18_19_13.361+09_00.csv': {'group': 'halfcheetah_gravity', 'env': 'PerturbHalfCheetah-v4', 'perturbation': 'gravity', 'rows': 15, 'state_counts': {'running': 15}}, 'wandb_export_2026-08-31T18_19_25.526+09_00.csv': {'group': 'halfcheetah_friction', 'env': 'PerturbHalfCheetah-v4', 'perturbation': 'friction', 'rows': 35, 'state_counts': {'running': 20, 'finished': 15}}, 'wandb_export_2026-08-31T18_19_36.155+09_00.csv': {'group': 'walker2d_friction', 'env': 'PerturbWalker2d-v4', 'perturbation': 'friction', 'rows': 15, 'state_counts': {'finished': 15}}}

# RUN_IDS_BY_GROUP = {
#     'hopper_friction': [
#         'xrte45g6',
#         '535nxqnd',
#         'oxhkpulo',
#         '4nr2t8zm',
#         'jogdn4bn',
#         'igwvhoci',
#         'sndqngi7',
#         'pxj33jw2',
#         '3ztlo13d',
#         '5be4r7rl',
#     ],
#     'hopper_gravity': [
#         '8od06e69',
#         '23rztu6t',
#         'kkod87jr',
#         '1n02xzxs',
#         'k2x7o90t',
#         'qo43bf6s',
#     ],
#     'ant_gravity': [
#         'qo7i6fll',
#         's5cqw7pq',
#         'a8a1u0m6',
#         'scktpqrq',
#         'xk4ftv2f',
#         'qc1aistw',
#         'yn952l4n',
#         'iuibv3kn',
#         'vf9rfobi',
#         'qysddk52',
#         '5ufm7ndd',
#         'ignh5lfe',
#         '5o829tv0',
#         'ngpnketj',
#         's9ux3zid',
#     ],
#     'ant_friction': [
#         '5gk6mlu2',
#         '8btm4ezm',
#         'v09m2pbi',
#         'bf3imy2m',
#         'q5gxe6r7',
#         'fj778472',
#         '02rzx00q',
#         'jlf4ki0a',
#         'h3dxbntu',
#         'p3enpd7s',
#         'fcz7wzak',
#         'qqtkbux0',
#         'sk3tddj9',
#         'y2wf93lo',
#         'c3vc2x2u',
#     ],
#     'halfcheetah_friction': [
#         'buwy599w',
#         'liabhch6',
#         'czju9k9g',
#         't4tlkmp0',
#         'qcf98mzq',
#         'mzip6j0a',
#         'xrtruq8v',
#         'fdqos87u',
#         '2ci3uapa',
#         'k578axsb',
#         '5b0pa4bw',
#         'e54yv3ju',
#         'ub3ptrm2',
#         '7isfodsn',
#         'y6hq7jlg',
#         'b4ycegsm',
#         '1htevbt3',
#         'azhvv3v2',
#         '9lzb745o',
#         'f8ldrlar',
#         '257omi2u',
#         'bmzieulj',
#         'p08x6vb4',
#         '0pl7ie7l',
#         '1mie8qn7',
#         'sij2zuht',
#         'h4smdalz',
#         'cusaxdbj',
#         'kw5cdkkw',
#         'sv6n20xp',
#         'p4iuycv6',
#         'zqu8a8cl',
#         '0y5ghyu7',
#         'x3mj8w9r',
#         'smfg99ll',
#     ],
#     'halfcheetah_gravity': [
#         'myemsjql',
#         '27tzdqkx',
#         'x84fcrg8',
#         'srqobu27',
#         'kokra7dm',
#         'dnrn2b1t',
#         'k0va83ir',
#         '9nexx9r6',
#         'q3si00b9',
#         'xa8boisn',
#         'suq43xd4',
#         'tbb2wlaf',
#         'hnr7d64e',
#         '763gi9ua',
#         'ki77184e',
#     ],
#     'walker2d_friction': [
#         'axtfjrx7',
#         'lmq5kiuv',
#         'hb30ng3h',
#         'd4oesiiv',
#         'qrjjyxzn',
#         'fxba195i',
#         'ww28q347',
#         'gqdspk3g',
#         'zh1lcu91',
#         '05b5a3gh',
#         'aj3s9tlu',
#         'nqgoxgwf',
#         'd6jm7gu3',
#         'dlfxco38',
#         'weoa5i1v',
#     ],
# }

# RUN_IDS = list(dict.fromkeys(
#     run_id
#     for ids in RUN_IDS_BY_GROUP.values()
#     for run_id in ids
# ))

# assert len(RUN_IDS) == 111, f"Expected 111 unique runs, got {len(RUN_IDS)}"


# def json_safe(x):
#     if x is None:
#         return None
#     if isinstance(x, float):
#         return None if math.isnan(x) or math.isinf(x) else x
#     if isinstance(x, (str, int, bool)):
#         return x
#     if isinstance(x, datetime):
#         return x.isoformat()
#     try:
#         import numpy as np
#         if isinstance(x, np.integer):
#             return int(x)
#         if isinstance(x, np.floating):
#             x = float(x)
#             return None if math.isnan(x) or math.isinf(x) else x
#         if isinstance(x, np.ndarray):
#             return x.tolist()
#     except Exception:
#         pass
#     if isinstance(x, dict):
#         return {str(k): json_safe(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [json_safe(v) for v in x]
#     return str(x)


# def dump_json(path: Path, obj):
#     path.write_text(
#         json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )


# def group_for_run(run_id: str):
#     for group, ids in RUN_IDS_BY_GROUP.items():
#         if run_id in ids:
#             return group
#     return "unknown"


# def export_run(run, out_root: Path):
#     run_id = str(run.id)
#     group = group_for_run(run_id)
#     out = out_root / group / run_id
#     out.mkdir(parents=True, exist_ok=True)

#     config = {
#         k: v for k, v in dict(run.config).items()
#         if not str(k).startswith("_")
#     }
#     dump_json(out / "config.json", config)

#     try:
#         summary = dict(run.summary)
#     except Exception:
#         try:
#             summary = run.summary._json_dict
#         except Exception:
#             summary = {}
#     dump_json(out / "summary.json", summary)

#     try:
#         api_path = "/".join(run.path)
#     except Exception:
#         api_path = f"{ENTITY}/{PROJECT}/{run_id}"

#     metadata = {
#         "experiment_group": group,
#         "api_path": api_path,
#         "entity": getattr(run, "entity", None),
#         "project": getattr(run, "project", None),
#         "id": getattr(run, "id", None),
#         "name": getattr(run, "name", None),
#         "state": getattr(run, "state", None),
#         "created_at": getattr(run, "created_at", None),
#         "url": getattr(run, "url", None),
#         "tags": getattr(run, "tags", None),
#         "notes": getattr(run, "notes", None),
#     }
#     dump_json(out / "metadata.json", metadata)

#     n_rows = 0
#     all_keys = set()
#     first_rows = []
#     last_rows = []

#     with (out / "history.jsonl").open("w", encoding="utf-8") as f:
#         for row in run.scan_history(page_size=PAGE_SIZE):
#             row = json_safe(row)
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")
#             n_rows += 1
#             all_keys.update(row.keys())
#             if len(first_rows) < 5:
#                 first_rows.append(row)
#             last_rows.append(row)
#             if len(last_rows) > 5:
#                 last_rows.pop(0)

#     readme = [
#         "# W&B Run Export", "",
#         f"- Experiment group: `{group}`",
#         f"- API path: `{api_path}`",
#         f"- Run name: `{getattr(run, 'name', None)}`",
#         f"- Run id: `{run_id}`",
#         f"- State at export: `{getattr(run, 'state', None)}`",
#         f"- History rows: `{n_rows}`", "",
#         "## Files",
#         "- `metadata.json`",
#         "- `config.json`",
#         "- `summary.json`",
#         "- `history.jsonl`", "",
#         "## Logged keys",
#         ", ".join(f"`{k}`" for k in sorted(all_keys)), "",
#         "## Summary", "```json",
#         json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
#         "```", "", "## Config", "```json",
#         json.dumps(json_safe(config), ensure_ascii=False, indent=2),
#         "```", "", "## First 5 history rows", "```json",
#         json.dumps(first_rows, ensure_ascii=False, indent=2),
#         "```", "", "## Last 5 history rows", "```json",
#         json.dumps(last_rows, ensure_ascii=False, indent=2),
#         "```",
#     ]
#     (out / "README_for_GPT.md").write_text("\n".join(readme), encoding="utf-8")

#     return {
#         "id": run_id,
#         "group": group,
#         "name": getattr(run, "name", None),
#         "state": getattr(run, "state", None),
#         "history_rows": n_rows,
#         "out_dir": str(out),
#     }


# def main():
#     api = wandb.Api()
#     out_root = Path(OUT_ROOT)
#     out_root.mkdir(parents=True, exist_ok=True)

#     print(f"[INFO] unique runs: {len(RUN_IDS)}")
#     for group, ids in RUN_IDS_BY_GROUP.items():
#         print(f"  - {group}: {len(ids)}")
#     print()

#     exported, skipped, errors = [], [], []

#     for i, run_id in enumerate(RUN_IDS, start=1):
#         group = group_for_run(run_id)
#         api_path = f"{ENTITY}/{PROJECT}/{run_id}"
#         print(f"[{i:03d}/{len(RUN_IDS):03d}] {group} / {run_id}")
#         try:
#             run = api.run(api_path)
#             state = str(getattr(run, "state", "")).lower()
#             if ONLY_FINISHED and state != "finished":
#                 print(f"    SKIP: state={state}")
#                 skipped.append({"id": run_id, "group": group, "state": state})
#                 continue
#             result = export_run(run, out_root)
#             exported.append(result)
#             print(f"    DONE: {result['history_rows']} history rows")
#         except Exception as e:
#             print(f"    ERROR: {e}")
#             errors.append({
#                 "id": run_id,
#                 "group": group,
#                 "api_path": api_path,
#                 "error": str(e),
#             })

#     dump_json(out_root / "_run_ids_by_group.json", RUN_IDS_BY_GROUP)
#     dump_json(out_root / "_source_file_metadata.json", SOURCE_FILE_METADATA)
#     dump_json(out_root / "_export_summary.json", {
#         "entity": ENTITY,
#         "project": PROJECT,
#         "unique_run_count": len(RUN_IDS),
#         "only_finished": ONLY_FINISHED,
#         "expected_by_group": {g: len(ids) for g, ids in RUN_IDS_BY_GROUP.items()},
#         "exported_count": len(exported),
#         "skipped_count": len(skipped),
#         "error_count": len(errors),
#         "exported": exported,
#         "skipped": skipped,
#         "errors": errors,
#     })

#     print()
#     print("=" * 72)
#     print(f"Expected : {len(RUN_IDS)}")
#     print(f"Exported : {len(exported)}")
#     print(f"Skipped  : {len(skipped)}")
#     print(f"Errors   : {len(errors)}")
#     print(f"Output   : {out_root}")
#     print("=" * 72)


# if __name__ == "__main__":
#     main()


# export_wandb_ant_no_cap_20_runs.py
#
# Add-on exporter for the EXISTING final-160 output directory.
#
# This script exports only the 20 Ant no-cap AMPO adaptive runs:
#
#   perturbations:
#     - gravity
#     - friction
#
#   variants:
#     - AMPO Adaptive no-cap, dual_lr=1e-4
#     - AMPO Adaptive no-cap, dual_lr=3e-4
#
#   seeds:
#     - 1, 2, 3, 4, 5
#
# Total = 2 perturbations * 2 dual_lr values * 5 seeds = 20 runs
#
# IMPORTANT:
#   The existing 160-run folders are NOT overwritten.
#   New folders are added under:
#
#   logs/wandb_logs_final_160/
#     ampo_adaptive_no_cap_dual_lr_1e-4/
#       ant/
#         gravity/
#         friction/
#     ampo_adaptive_no_cap_dual_lr_3e-4/
#       ant/
#         gravity/
#         friction/
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_ant_no_cap_20_runs.py

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import wandb


DEFAULT_ENTITY = "ukjo19"
DEFAULT_PROJECT = "sb3"

# Same root folder as the previous final-160 exporter.
DEFAULT_OUT_ROOT = "logs/wandb_logs_final_160"

DEFAULT_PAGE_SIZE = 500
DEFAULT_RETRIES = 3


RUNS = [{'id': 'jzxom1yp',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 3,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'rae0ing0',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 5,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'wm6udlw5',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 4,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'bbzqeaxf',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 5,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'lxmc4o6o',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 4,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'jkj3ka8n',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 3,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': '5z2n2cqq',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 2,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'bvq4i8ml',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 2,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'lhg7ww3o',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 1,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': '11qb4wd0',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 1,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 's97dxo52',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 5,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'wwcw5b1m',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 4,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'sshntafo',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 5,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'mfp1mpom',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 3,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': '8z4rle81',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 4,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'hamen8uh',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 2,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'zmx1bg71',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 3,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'yh4t7kg6',
  'variant': 'ampo_adaptive_no_cap_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 1,
  'dual_lr': 0.0003,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'feyisa8t',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 2,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'},
 {'id': 'wjrw7gc2',
  'variant': 'ampo_adaptive_no_cap_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 1,
  'dual_lr': 0.0001,
  'no_cap': True,
  'csv_state': 'finished'}]


def validate_hardcoded_grid():
    assert len(RUNS) == 20, f"Expected 20 runs, got {len(RUNS)}"

    ids = [r["id"] for r in RUNS]
    assert len(set(ids)) == 20, "Duplicate run IDs found"

    variants = [
        "ampo_adaptive_no_cap_dual_lr_1e-4",
        "ampo_adaptive_no_cap_dual_lr_3e-4",
    ]

    for variant in variants:
        for perturbation in ["gravity", "friction"]:
            seeds = sorted(
                r["seed"]
                for r in RUNS
                if r["variant"] == variant
                and r["perturbation"] == perturbation
            )
            assert seeds == [1, 2, 3, 4, 5], (
                variant,
                perturbation,
                seeds,
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


def run_output_dir(out_root: Path, spec: dict) -> Path:
    return (
        out_root
        / spec["variant"]
        / "ant"
        / spec["perturbation"]
        / f"seed_{spec['seed']}__{spec['id']}"
    )


def export_one_run(
    api,
    spec: dict,
    entity: str,
    project: str,
    out_root: Path,
    page_size: int,
    retries: int,
    skip_existing: bool,
):
    run_id = spec["id"]
    api_path = f"{entity}/{project}/{run_id}"

    out = run_output_dir(out_root, spec)
    success_marker = out / "_SUCCESS.json"

    if skip_existing and success_marker.exists():
        print("    SKIP: already exported")
        return {
            "status": "already_exported",
            **spec,
            "out_dir": str(out),
        }

    out.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            run = api.run(api_path)
            state = str(getattr(run, "state", "")).lower()

            # These 20 runs were all finished in the supplied CSV.
            if state != "finished":
                return {
                    "status": "skipped_not_finished",
                    **spec,
                    "api_state": state,
                }

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

            metadata = {
                "hardcoded_spec": spec,
                "experiment_note": "Ant AMPO adaptive no-cap add-on",
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

            with (out / "history.jsonl").open(
                "w",
                encoding="utf-8",
            ) as f:
                for row in run.scan_history(page_size=page_size):
                    row = json_safe(row)

                    f.write(
                        json.dumps(row, ensure_ascii=False)
                        + "\n"
                    )

                    n_rows += 1
                    all_keys.update(row.keys())

                    if len(first_rows) < 5:
                        first_rows.append(row)

                    last_rows.append(row)
                    if len(last_rows) > 5:
                        last_rows.pop(0)

            readme = [
                "# W&B Ant No-Cap Run Export",
                "",
                "- Experiment: `AMPO adaptive no-cap`",
                f"- Variant: `{spec['variant']}`",
                "- Environment: `ant`",
                f"- Perturbation: `{spec['perturbation']}`",
                f"- Seed: `{spec['seed']}`",
                f"- Dual LR: `{spec['dual_lr']}`",
                f"- API path: `{api_path}`",
                f"- Run name: `{getattr(run, 'name', None)}`",
                f"- Run id: `{run_id}`",
                f"- State: `{state}`",
                f"- History rows: `{n_rows}`",
                "",
                "## Files",
                "- `metadata.json`",
                "- `config.json`",
                "- `summary.json`",
                "- `history.jsonl`",
                "- `_SUCCESS.json`",
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

            result = {
                "status": "exported",
                **spec,
                "api_state": state,
                "history_rows": n_rows,
                "out_dir": str(out),
            }

            dump_json(success_marker, result)
            return result

        except Exception as e:
            last_error = e

            print(
                f"    attempt {attempt}/{retries} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))

    return {
        "status": "error",
        **spec,
        "error": (
            f"{type(last_error).__name__}: "
            f"{last_error}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add 20 Ant AMPO adaptive no-cap runs "
            "to the existing final-160 W&B log folder."
        )
    )

    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
    )

    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
    )

    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_ROOT,
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-export even when _SUCCESS.json "
            "already exists."
        ),
    )

    args = parser.parse_args()

    validate_hardcoded_grid()

    api = wandb.Api()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Separate manifest/summary so the original 160-run files
    # are not overwritten.
    dump_json(
        out_root / "_hardcoded_ant_no_cap_20_run_manifest.json",
        RUNS,
    )

    print("=" * 78)
    print("ANT NO-CAP ADD-ON EXPORT: 20 runs")
    print("  adaptive dual_lr=1e-4 : 10")
    print("  adaptive dual_lr=3e-4 : 10")
    print("  gravity                : 10")
    print("  friction               : 10")
    print(f"  output root            : {out_root}")
    print("=" * 78)
    print()

    ordered_runs = sorted(
        RUNS,
        key=lambda r: (
            r["variant"],
            r["perturbation"],
            r["seed"],
        ),
    )

    results = []

    for i, spec in enumerate(ordered_runs, start=1):
        print(
            f"[{i:02d}/20] "
            f"{spec['variant']} / "
            f"ant / "
            f"{spec['perturbation']} / "
            f"seed={spec['seed']} / "
            f"id={spec['id']}"
        )

        result = export_one_run(
            api=api,
            spec=spec,
            entity=args.entity,
            project=args.project,
            out_root=out_root,
            page_size=args.page_size,
            retries=max(1, args.retries),
            skip_existing=not args.force,
        )

        results.append(result)

        if result["status"] == "exported":
            print(
                f"    DONE: "
                f"{result['history_rows']} history rows"
            )

        elif result["status"] == "skipped_not_finished":
            print(
                f"    SKIP: state="
                f"{result.get('api_state')}"
            )

        elif result["status"] == "error":
            print(
                f"    ERROR: "
                f"{result.get('error')}"
            )

    counts = {}
    for item in results:
        counts[item["status"]] = (
            counts.get(item["status"], 0) + 1
        )

    dump_json(
        out_root / "_ant_no_cap_export_summary.json",
        {
            "expected_total": 20,
            "status_counts": counts,
            "results": results,
        },
    )

    print()
    print("=" * 78)
    print("ANT NO-CAP EXPORT SUMMARY")
    for status, count in sorted(counts.items()):
        print(f"  {status:24s} {count}")
    print(f"  output: {out_root}")
    print("=" * 78)


if __name__ == "__main__":
    main()
