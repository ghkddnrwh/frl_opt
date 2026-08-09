# export_finished_wandb_runs.py

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import wandb


# W&B export CSV에서 읽어온 50개 run ID.
# 실행 시 CSV 파일은 필요하지 않습니다.
RUN_IDS = [
    "snzhj71d",
    "spjz4yrx",
    "k4jl6z4w",
    "efv0o5h4",
    "0yn6b3n2",
    "c74roexc",
    "9rh59t1e",
    "3wo65je4",
    "uhmmfw6g",
    "04q97bk6",
    "lbm06ds1",
    "0l1iux5l",
    "2l56mzmy",
    "ybd5xu9s",
    "ct9gjytk",
    "khms6jp0",
    "dmx8peqv",
    "w0lqxvu7",
    "0r9vckmh",
    "8zbixsuq",
    "ntuoyds9",
    "yoatayjc",
    "7m37ol3l",
    "2cvuib45",
    "suydkk23",
    "s9dowb0q",
    "sxchguz7",
    "pqrtizfv",
    "4q721ygv",
    "ucv7g3qd",
    "c1g6zd56",
    "uyo7yq10",
    "z9vcjiv8",
    "mne8f20c",
    "zbf2uj59",
    "iif4sbyv",
    "l3wn708r",
    "kn0e5sw3",
    "orfq7bvq",
    "wa32p4zd",
    "c94sidk6",
    "8qkrgqyn",
    "qz2sw3d7",
    "94vhyfwj",
    "5qzv0ywu",
    "lpjaihgq",
    "053li8bw",
    "y2wjbzur",
    "vvyxyp2a",
    "75b9z5am",
]


def json_safe(x):
    """NaN, numpy scalar, datetime 등을 JSON-safe 값으로 변환."""
    if x is None:
        return None

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x

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


def export_wandb_run(
    run,
    out_root: str = "logs/wandb_logs",
    page_size: int = 500,
):
    """
    이미 조회한 wandb.apis.public.Run 객체 하나를 export한다.

    저장 구조:
      logs/wandb_logs/<run_id>/
        metadata.json
        config.json
        summary.json
        history.jsonl
        README_for_GPT.md
    """
    run_id = str(run.id)
    api_path = "/".join(run.path)

    out = Path(out_root) / run_id
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
        summary = getattr(run.summary, "_json_dict", {})
    dump_json(out / "summary.json", summary)

    # 3. metadata
    metadata = {
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

    # 4. 전체 scalar history -> JSONL
    history_path = out / "history.jsonl"

    n_rows = 0
    all_keys = set()
    first_rows = []
    last_rows = []

    with history_path.open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=page_size):
            row = json_safe(row)

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_rows += 1
            all_keys.update(row.keys())

            if len(first_rows) < 5:
                first_rows.append(row)

            last_rows.append(row)
            if len(last_rows) > 5:
                last_rows.pop(0)

    # 5. GPT/사람이 보기 쉬운 README
    md = []
    md.append("# W&B Run Export")
    md.append("")
    md.append(f"- API path: `{api_path}`")
    md.append(f"- Run name: `{getattr(run, 'name', None)}`")
    md.append(f"- Run id: `{getattr(run, 'id', None)}`")
    md.append(f"- State: `{getattr(run, 'state', None)}`")
    md.append(f"- Created at: `{getattr(run, 'created_at', None)}`")
    md.append(f"- URL: {getattr(run, 'url', None)}")
    md.append("")
    md.append("## Files")
    md.append("- `metadata.json`: run 기본 정보")
    md.append("- `config.json`: 하이퍼파라미터")
    md.append("- `summary.json`: 최종 metric / summary")
    md.append("- `history.jsonl`: 전체 step별 로그")
    md.append("")
    md.append("## History")
    md.append(f"- Number of rows: `{n_rows}`")
    md.append(f"- Number of keys: `{len(all_keys)}`")
    md.append("")
    md.append("### Logged keys")
    md.append(", ".join(f"`{k}`" for k in sorted(all_keys)))
    md.append("")
    md.append("## Summary")
    md.append("```json")
    md.append(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## Config")
    md.append("```json")
    md.append(json.dumps(json_safe(config), ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## First 5 history rows")
    md.append("```json")
    md.append(json.dumps(json_safe(first_rows), ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## Last 5 history rows")
    md.append("```json")
    md.append(json.dumps(json_safe(last_rows), ensure_ascii=False, indent=2))
    md.append("```")

    (out / "README_for_GPT.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print(f"[DONE] {run_id} -> {out}")
    return {
        "id": run_id,
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "rows": n_rows,
        "out_dir": str(out),
    }


def export_finished_runs(
    entity: str,
    project: str,
    out_root: str = "logs/wandb_logs",
    page_size: int = 500,
    expected_finished: int | None = 30,
):
    """
    하드코딩된 RUN_IDS를 W&B API에서 조회한 뒤,
    state == 'finished'인 run만 export한다.
    """
    api = wandb.Api()

    run_ids = RUN_IDS
    print(f"[INFO] 하드코딩된 run 수: {len(run_ids)}")

    finished_runs = []
    other_runs = []
    lookup_errors = []

    # 먼저 상태를 전부 확인한다.
    for i, run_id in enumerate(run_ids, start=1):
        api_path = f"{entity}/{project}/{run_id}"

        try:
            run = api.run(api_path)
            state = str(getattr(run, "state", "")).lower()

            print(
                f"[CHECK {i:02d}/{len(run_ids):02d}] "
                f"{run_id}  state={state}"
            )

            if state == "finished":
                finished_runs.append(run)
            else:
                other_runs.append(
                    {
                        "id": run_id,
                        "state": state,
                        "name": getattr(run, "name", None),
                    }
                )

        except Exception as e:
            print(f"[ERROR] run 조회 실패: {api_path}: {e}")
            lookup_errors.append(
                {
                    "id": run_id,
                    "api_path": api_path,
                    "error": str(e),
                }
            )

    print()
    print(f"[INFO] finished run 수: {len(finished_runs)}")
    print(f"[INFO] finished 이외 run 수: {len(other_runs)}")
    print(f"[INFO] 조회 실패 run 수: {len(lookup_errors)}")

    if expected_finished is not None and len(finished_runs) != expected_finished:
        print(
            f"[WARNING] 예상 finished={expected_finished}개였지만 "
            f"실제 API 조회 결과는 {len(finished_runs)}개입니다."
        )

    # 상태 확인 결과 저장
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    dump_json(
        out_root_path / "_run_status.json",
        {
            "entity": entity,
            "project": project,
            "total_run_ids": len(run_ids),
            "finished_count": len(finished_runs),
            "other_count": len(other_runs),
            "lookup_error_count": len(lookup_errors),
            "finished_ids": [run.id for run in finished_runs],
            "other_runs": other_runs,
            "lookup_errors": lookup_errors,
        },
    )

    # finished run만 실제 history export
    exported = []
    export_errors = []

    for i, run in enumerate(finished_runs, start=1):
        print()
        print(
            f"[EXPORT {i:02d}/{len(finished_runs):02d}] "
            f"{run.id} ({run.name})"
        )

        try:
            result = export_wandb_run(
                run=run,
                out_root=out_root,
                page_size=page_size,
            )
            exported.append(result)

        except Exception as e:
            print(f"[ERROR] export 실패: {run.id}: {e}")
            export_errors.append(
                {
                    "id": run.id,
                    "name": getattr(run, "name", None),
                    "error": str(e),
                }
            )

    # 전체 batch 결과 저장
    dump_json(
        out_root_path / "_export_summary.json",
        {
            "total_run_ids": len(run_ids),
            "finished_detected": len(finished_runs),
            "successfully_exported": len(exported),
            "export_error_count": len(export_errors),
            "exported": exported,
            "export_errors": export_errors,
        },
    )

    print()
    print("=" * 70)
    print(f"[COMPLETE] finished detected : {len(finished_runs)}")
    print(f"[COMPLETE] exported          : {len(exported)}")
    print(f"[COMPLETE] export errors     : {len(export_errors)}")
    print(f"[COMPLETE] output root       : {out_root_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="하드코딩된 W&B run ID 중 finished 상태인 run만 전체 history export"
    )
    parser.add_argument(
        "--entity",
        default="ukjo19",
        help="W&B entity (default: ukjo19)",
    )
    parser.add_argument(
        "--project",
        default="sb3",
        help="W&B project (default: sb3)",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/wandb_logs",
        help="출력 root directory",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="scan_history page size",
    )
    parser.add_argument(
        "--expected-finished",
        type=int,
        default=30,
        help="예상 finished run 수. 다르면 warning 출력",
    )

    args = parser.parse_args()

    export_finished_runs(
        entity=args.entity,
        project=args.project,
        out_root=args.out_dir,
        page_size=args.page_size,
        expected_finished=args.expected_finished,
    )


if __name__ == "__main__":
    main()
