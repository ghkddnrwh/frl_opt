# export_wandb_runs_2026_08_31_hardcoded.py
# Generated from 9 W&B CSV exports.
# CSV files are not needed at runtime; all unique run IDs are hardcoded.
# Raw CSV rows: 161
# Unique runs: 111
# Duplicate occurrences removed: 50
# CSV state counts before deduplication: {'finished': 105, 'running': 56}

import json
import math
from datetime import datetime
from pathlib import Path
import wandb

ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_08_31"
PAGE_SIZE = 500

# False: export every hardcoded run, including currently running ones.
# True: export only runs whose current W&B API state is finished.
ONLY_FINISHED = False

SOURCE_FILE_METADATA = {'wandb_export_2026-08-31T18_10_23.444+09_00.csv': {'group': 'hopper_friction', 'env': 'PerturbHopper-v4', 'perturbation': 'friction', 'rows': 10, 'state_counts': {'finished': 10}}, 'wandb_export_2026-08-31T18_10_41.973+09_00.csv': {'group': 'hopper_gravity', 'env': 'PerturbHopper-v4', 'perturbation': 'gravity', 'rows': 6, 'state_counts': {'finished': 5, 'running': 1}}, 'wandb_export_2026-08-31T18_10_56.441+09_00.csv': {'group': 'ant_gravity', 'env': 'PerturbAnt-v4', 'perturbation': 'gravity', 'rows': 15, 'state_counts': {'finished': 15}}, 'wandb_export_2026-08-31T18_15_33.077+09_00.csv': {'group': 'ant_gravity', 'env': 'PerturbAnt-v4', 'perturbation': 'gravity', 'rows': 15, 'state_counts': {'finished': 15}}, 'wandb_export_2026-08-31T18_15_43.721+09_00.csv': {'group': 'ant_friction', 'env': 'PerturbAnt-v4', 'perturbation': 'friction', 'rows': 15, 'state_counts': {'finished': 15}}, 'wandb_export_2026-08-31T18_15_55.481+09_00.csv': {'group': 'halfcheetah_friction', 'env': 'PerturbHalfCheetah-v4', 'perturbation': 'friction', 'rows': 35, 'state_counts': {'running': 20, 'finished': 15}}, 'wandb_export_2026-08-31T18_19_13.361+09_00.csv': {'group': 'halfcheetah_gravity', 'env': 'PerturbHalfCheetah-v4', 'perturbation': 'gravity', 'rows': 15, 'state_counts': {'running': 15}}, 'wandb_export_2026-08-31T18_19_25.526+09_00.csv': {'group': 'halfcheetah_friction', 'env': 'PerturbHalfCheetah-v4', 'perturbation': 'friction', 'rows': 35, 'state_counts': {'running': 20, 'finished': 15}}, 'wandb_export_2026-08-31T18_19_36.155+09_00.csv': {'group': 'walker2d_friction', 'env': 'PerturbWalker2d-v4', 'perturbation': 'friction', 'rows': 15, 'state_counts': {'finished': 15}}}

RUN_IDS_BY_GROUP = {
    'hopper_friction': [
        'xrte45g6',
        '535nxqnd',
        'oxhkpulo',
        '4nr2t8zm',
        'jogdn4bn',
        'igwvhoci',
        'sndqngi7',
        'pxj33jw2',
        '3ztlo13d',
        '5be4r7rl',
    ],
    'hopper_gravity': [
        '8od06e69',
        '23rztu6t',
        'kkod87jr',
        '1n02xzxs',
        'k2x7o90t',
        'qo43bf6s',
    ],
    'ant_gravity': [
        'qo7i6fll',
        's5cqw7pq',
        'a8a1u0m6',
        'scktpqrq',
        'xk4ftv2f',
        'qc1aistw',
        'yn952l4n',
        'iuibv3kn',
        'vf9rfobi',
        'qysddk52',
        '5ufm7ndd',
        'ignh5lfe',
        '5o829tv0',
        'ngpnketj',
        's9ux3zid',
    ],
    'ant_friction': [
        '5gk6mlu2',
        '8btm4ezm',
        'v09m2pbi',
        'bf3imy2m',
        'q5gxe6r7',
        'fj778472',
        '02rzx00q',
        'jlf4ki0a',
        'h3dxbntu',
        'p3enpd7s',
        'fcz7wzak',
        'qqtkbux0',
        'sk3tddj9',
        'y2wf93lo',
        'c3vc2x2u',
    ],
    'halfcheetah_friction': [
        'buwy599w',
        'liabhch6',
        'czju9k9g',
        't4tlkmp0',
        'qcf98mzq',
        'mzip6j0a',
        'xrtruq8v',
        'fdqos87u',
        '2ci3uapa',
        'k578axsb',
        '5b0pa4bw',
        'e54yv3ju',
        'ub3ptrm2',
        '7isfodsn',
        'y6hq7jlg',
        'b4ycegsm',
        '1htevbt3',
        'azhvv3v2',
        '9lzb745o',
        'f8ldrlar',
        '257omi2u',
        'bmzieulj',
        'p08x6vb4',
        '0pl7ie7l',
        '1mie8qn7',
        'sij2zuht',
        'h4smdalz',
        'cusaxdbj',
        'kw5cdkkw',
        'sv6n20xp',
        'p4iuycv6',
        'zqu8a8cl',
        '0y5ghyu7',
        'x3mj8w9r',
        'smfg99ll',
    ],
    'halfcheetah_gravity': [
        'myemsjql',
        '27tzdqkx',
        'x84fcrg8',
        'srqobu27',
        'kokra7dm',
        'dnrn2b1t',
        'k0va83ir',
        '9nexx9r6',
        'q3si00b9',
        'xa8boisn',
        'suq43xd4',
        'tbb2wlaf',
        'hnr7d64e',
        '763gi9ua',
        'ki77184e',
    ],
    'walker2d_friction': [
        'axtfjrx7',
        'lmq5kiuv',
        'hb30ng3h',
        'd4oesiiv',
        'qrjjyxzn',
        'fxba195i',
        'ww28q347',
        'gqdspk3g',
        'zh1lcu91',
        '05b5a3gh',
        'aj3s9tlu',
        'nqgoxgwf',
        'd6jm7gu3',
        'dlfxco38',
        'weoa5i1v',
    ],
}

RUN_IDS = list(dict.fromkeys(
    run_id
    for ids in RUN_IDS_BY_GROUP.values()
    for run_id in ids
))

assert len(RUN_IDS) == 111, f"Expected 111 unique runs, got {len(RUN_IDS)}"


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
        k: v for k, v in dict(run.config).items()
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
        "# W&B Run Export", "",
        f"- Experiment group: `{group}`",
        f"- API path: `{api_path}`",
        f"- Run name: `{getattr(run, 'name', None)}`",
        f"- Run id: `{run_id}`",
        f"- State at export: `{getattr(run, 'state', None)}`",
        f"- History rows: `{n_rows}`", "",
        "## Files",
        "- `metadata.json`",
        "- `config.json`",
        "- `summary.json`",
        "- `history.jsonl`", "",
        "## Logged keys",
        ", ".join(f"`{k}`" for k in sorted(all_keys)), "",
        "## Summary", "```json",
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        "```", "", "## Config", "```json",
        json.dumps(json_safe(config), ensure_ascii=False, indent=2),
        "```", "", "## First 5 history rows", "```json",
        json.dumps(first_rows, ensure_ascii=False, indent=2),
        "```", "", "## Last 5 history rows", "```json",
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

    print(f"[INFO] unique runs: {len(RUN_IDS)}")
    for group, ids in RUN_IDS_BY_GROUP.items():
        print(f"  - {group}: {len(ids)}")
    print()

    exported, skipped, errors = [], [], []

    for i, run_id in enumerate(RUN_IDS, start=1):
        group = group_for_run(run_id)
        api_path = f"{ENTITY}/{PROJECT}/{run_id}"
        print(f"[{i:03d}/{len(RUN_IDS):03d}] {group} / {run_id}")
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

    dump_json(out_root / "_run_ids_by_group.json", RUN_IDS_BY_GROUP)
    dump_json(out_root / "_source_file_metadata.json", SOURCE_FILE_METADATA)
    dump_json(out_root / "_export_summary.json", {
        "entity": ENTITY,
        "project": PROJECT,
        "unique_run_count": len(RUN_IDS),
        "only_finished": ONLY_FINISHED,
        "expected_by_group": {g: len(ids) for g, ids in RUN_IDS_BY_GROUP.items()},
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "exported": exported,
        "skipped": skipped,
        "errors": errors,
    })

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
