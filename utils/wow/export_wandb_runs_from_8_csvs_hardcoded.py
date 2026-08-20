# export_wandb_runs_2026_08_20_hardcoded.py

import json
import math
from datetime import datetime
from pathlib import Path
import wandb

ENTITY = "ukjo19"
PROJECT = "sb3"
OUT_ROOT = "logs/wandb_logs_2026_08_20"
PAGE_SIZE = 500
ONLY_FINISHED = False

RUN_IDS_BY_SOURCE = {
    'wandb_export_2026-08-20T14_29_10.212+09_00.csv': [
        '7x752p6o',
        'yriqxvsl',
        'c82aopab',
        'vwxtxkcv',
        'c8ri6khm',
        'q6e1f3v2',
        '46ilimta',
        'wlt4gli4',
        '5eerr4zt',
        'gpxjosag',
        'phwdcdfa',
        'hvxn590g',
        'fgn51v6r',
        'go9y73x2',
        'i7zd7bh1',
        'q0la1gr6',
        '1h23t57p',
        'umiuf4v7',
        'p55w6diu',
        'nd73cyed',
        'ofzhowes',
        'fb5oiu0u',
        'wpicdnmk',
        '2wyx2ubi',
        '9hp73etv',
        'qt38gy4q',
        '5r0qqtf2',
        '30427ipx',
        'pnn48cte',
        'avczqhdv',
    ],
    'wandb_export_2026-08-20T14_29_30.492+09_00.csv': [
        '0gxuusaz',
        'dl771pgy',
        '4c0ax3cp',
        'lbj6a7u8',
        'bbbm9nxb',
        'p0yo9kzo',
        'u53ytznz',
        'udq1bze8',
        '5pm9t06u',
        'aa976k8a',
        'my9eboxc',
        'c229yunk',
        'fj4xljvy',
        'k135v7oi',
        '9roc0rmb',
        'y32jwx76',
        'o5n3qthn',
        'ht7i4n1o',
        'g6iqd8om',
        '1388qfjq',
        '4o8kswbq',
        '5h32e6o3',
        'zo1pdp3j',
        'lzpf0lsg',
        'j44dt0zw',
        'phpje30m',
        'n0unown7',
        'ywdjz03q',
        'nfgcqo0x',
        '14pcwiak',
    ],
    'wandb_export_2026-08-20T14_29_44.640+09_00.csv': [
        'voerqy8v',
        'oywbyprk',
        'f1s2vpyw',
        'e2pn72hy',
        'osdhyn2h',
        '7ovd8hvk',
        'h0bv29y1',
        '7f27jefd',
        'xkkfci0d',
        'q4pupfv9',
        'iqmp04qi',
        'slutiey4',
        'ynqgha1l',
        '7tqun84p',
        '0om3c9eq',
        'cvom3w3r',
        '7f60a0wq',
        'xfcdj7mg',
        '33f324oy',
        'ohltkvv1',
    ],
    'wandb_export_2026-08-20T14_29_59.123+09_00.csv': [
        'hf8c8nty',
        '86ea07b1',
        '7591plmo',
        'iij01a2b',
        'j0oc5038',
        '2vifxoix',
        'b3ojx662',
        'o8a296ic',
        '88rorwx6',
        '28txr5w2',
        '5gk31hrv',
        '81sqrzyd',
        't0hyb5io',
        'tdbiui5t',
        's96kw76j',
        'ahbdxffi',
        'hg2ytg9l',
        'snfw0rmi',
        'wl7rby1o',
        '6aj2mkea',
    ],
    'wandb_export_2026-08-20T14_32_08.457+09_00.csv': [
        'bej7l6lc',
        '0cc099lj',
        '0fd7ebq7',
        'ynv2v2wm',
        'byln9tvl',
        'gwqew0k7',
        'flonuuw7',
        'utl40jdw',
        'l7gmxukk',
        'ulop4pfy',
    ],
    'wandb_export_2026-08-20T14_32_24.222+09_00.csv': [
        '83h9yt4b',
        'rmdybum0',
        'ii4j88th',
        '2y8td6vx',
        'i9v31d5f',
        'mzzov6bb',
        '8fcoe118',
        'fpxlbi33',
        'yr6vjkia',
        'quzkzfbt',
    ],
}

RUN_IDS = list(dict.fromkeys(run_id for ids in RUN_IDS_BY_SOURCE.values() for run_id in ids))

def json_safe(x):
    if x is None: return None
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, (str, int, bool)): return x
    if isinstance(x, datetime): return x.isoformat()
    try:
        import numpy as np
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating):
            x=float(x); return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(x, np.ndarray): return x.tolist()
    except Exception: pass
    if isinstance(x, dict): return {str(k): json_safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    return str(x)

def dump_json(path, obj):
    Path(path).write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")

def source_csvs_for_run(run_id):
    return [src for src, ids in RUN_IDS_BY_SOURCE.items() if run_id in ids]

def export_run(run, out_root):
    run_id=str(run.id)
    out=Path(out_root)/run_id
    out.mkdir(parents=True, exist_ok=True)
    try: api_path="/".join(run.path)
    except Exception: api_path=f"{ENTITY}/{PROJECT}/{run_id}"
    config={k:v for k,v in dict(run.config).items() if not str(k).startswith("_")}
    dump_json(out/"config.json", config)
    try: summary=dict(run.summary)
    except Exception:
        try: summary=run.summary._json_dict
        except Exception: summary={}
    dump_json(out/"summary.json", summary)
    metadata={"source_csvs":source_csvs_for_run(run_id),"api_path":api_path,"entity":getattr(run,"entity",None),"project":getattr(run,"project",None),"id":getattr(run,"id",None),"name":getattr(run,"name",None),"state":getattr(run,"state",None),"created_at":getattr(run,"created_at",None),"url":getattr(run,"url",None),"tags":getattr(run,"tags",None),"notes":getattr(run,"notes",None)}
    dump_json(out/"metadata.json", metadata)
    n_rows=0; all_keys=set(); first_rows=[]; last_rows=[]
    with (out/"history.jsonl").open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=PAGE_SIZE):
            row=json_safe(row); f.write(json.dumps(row, ensure_ascii=False)+"\n")
            n_rows+=1; all_keys.update(row.keys())
            if len(first_rows)<5: first_rows.append(row)
            last_rows.append(row)
            if len(last_rows)>5: last_rows.pop(0)
    md=["# W&B Run Export","",f"- Source CSV(s): `{source_csvs_for_run(run_id)}`",f"- API path: `{api_path}`",f"- Run name: `{getattr(run, 'name', None)}`",f"- Run id: `{run_id}`",f"- State at export: `{getattr(run, 'state', None)}`",f"- Created at: `{getattr(run, 'created_at', None)}`",f"- URL: {getattr(run, 'url', None)}","","## Files","- `metadata.json`: run 기본 정보","- `config.json`: 하이퍼파라미터","- `summary.json`: 현재/최종 summary","- `history.jsonl`: 전체 step별 scalar history","","## History",f"- Number of rows: `{n_rows}`",f"- Number of keys: `{len(all_keys)}`","","### Logged keys",", ".join(f"`{k}`" for k in sorted(all_keys)),"","## Summary","```json",json.dumps(json_safe(summary), ensure_ascii=False, indent=2),"```","","## Config","```json",json.dumps(json_safe(config), ensure_ascii=False, indent=2),"```","","## First 5 history rows","```json",json.dumps(json_safe(first_rows), ensure_ascii=False, indent=2),"```","","## Last 5 history rows","```json",json.dumps(json_safe(last_rows), ensure_ascii=False, indent=2),"```"]
    (out/"README_for_GPT.md").write_text("\n".join(md), encoding="utf-8")
    return {"id":run_id,"name":getattr(run,"name",None),"state":getattr(run,"state",None),"source_csvs":source_csvs_for_run(run_id),"history_rows":n_rows,"out_dir":str(out)}

def main():
    api=wandb.Api(); out_root=Path(OUT_ROOT); out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] entity/project : {ENTITY}/{PROJECT}")
    print(f"[INFO] hardcoded runs : {len(RUN_IDS)}")
    print(f"[INFO] only finished  : {ONLY_FINISHED}")
    print(f"[INFO] output         : {out_root}\n")
    exported=[]; skipped=[]; errors=[]
    for i,run_id in enumerate(RUN_IDS,1):
        api_path=f"{ENTITY}/{PROJECT}/{run_id}"
        print(f"[{i:03d}/{len(RUN_IDS):03d}] {api_path}")
        try:
            run=api.run(api_path); state=str(getattr(run,"state","")).lower()
            print(f"    name={getattr(run, 'name', None)} state={state}")
            if ONLY_FINISHED and state != "finished":
                skipped.append({"id":run_id,"state":state,"source_csvs":source_csvs_for_run(run_id)}); print("    -> SKIP (not finished)"); continue
            result=export_run(run,out_root); exported.append(result); print(f"    -> DONE (history rows={result['history_rows']})")
        except Exception as e:
            errors.append({"id":run_id,"api_path":api_path,"source_csvs":source_csvs_for_run(run_id),"error":str(e)}); print(f"    -> ERROR: {e}")
    dump_json(out_root/"_export_summary.json", {"entity":ENTITY,"project":PROJECT,"hardcoded_run_count":len(RUN_IDS),"only_finished":ONLY_FINISHED,"exported_count":len(exported),"skipped_count":len(skipped),"error_count":len(errors),"exported":exported,"skipped":skipped,"errors":errors})
    dump_json(out_root/"_run_ids_by_source.json", RUN_IDS_BY_SOURCE)
    print("\n"+"="*72)
    print(f"[COMPLETE] hardcoded runs : {len(RUN_IDS)}")
    print(f"[COMPLETE] exported       : {len(exported)}")
    print(f"[COMPLETE] skipped        : {len(skipped)}")
    print(f"[COMPLETE] errors         : {len(errors)}")
    print(f"[COMPLETE] output         : {out_root}")
    print("="*72)

if __name__ == "__main__": main()
