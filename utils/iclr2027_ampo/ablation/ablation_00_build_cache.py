from __future__ import annotations

from ampo_ablation_common import (
    apply_publication_style,
    build_run_cache,
    common_parser,
    discover_runs,
    validate_expected_layout,
)


def main() -> None:
    parser = common_parser("Build reduced caches for the 160-run AMPO ablation archive.")
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)
    cache_root = args.out_root / "_cache"
    runs = discover_runs(args.log_root)
    for idx, spec in enumerate(runs, start=1):
        build_run_cache(spec, cache_root, force=args.force_cache)
        if idx % 10 == 0 or idx == len(runs):
            print(f"[Progress] {idx}/{len(runs)} runs cached")
    print(f"[Done] {cache_root}")


if __name__ == "__main__":
    main()
