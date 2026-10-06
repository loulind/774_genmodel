"""Command line: one command per dataset.

    genagent run --data data/pathmnist.npz --objective "..." --out outputs/generated_report_1
    genagent run ... --quick          # smoke test: seconds per model
    genagent run ... --no-llm         # rule-based planner, critic and writer (no LLM access)
    genagent run ... --replay runs/<run>/llm_cache   # reproduce LLM decisions from cache
    genagent hardware                 # report detected device and budgets
"""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="genagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the agent on one dataset and objective")
    r.add_argument("--data", required=True, help=".npz, .h5ad, .csv/.tsv or .npy file")
    r.add_argument("--objective", required=True, help="free-text objective, in quotes")
    r.add_argument("--out", required=True, help="output directory for the report and artefacts")
    r.add_argument("--config", help="YAML with budgets/seed/LLM settings (never model choices)")
    r.add_argument("--quick", action="store_true", help="smoke-test budgets")
    r.add_argument("--no-llm", action="store_true", help="rule-based planner/critic/writer only")
    r.add_argument("--replay", help="directory of cached LLM responses to replay")
    r.add_argument("--label-column", help="label column name (tables / AnnData obs)")
    r.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--seed", type=int)
    r.add_argument("--model", help="LLM model id (default from config)")
    sub.add_parser("hardware", help="print detected hardware and budgets")
    args = ap.parse_args(argv)

    if args.cmd == "hardware":
        from .config import load_config
        from .hardware import detect

        hw = detect()
        cfg = load_config()
        print(json.dumps({"hardware": hw.to_record(),
                          "full_budget": cfg.budget(hw.device).model_dump(),
                          "quick_budget": load_config(quick=True).budget(hw.device).model_dump()},
                         indent=2))
        return

    from .orchestrator import run

    overrides = {"quick": args.quick, "no_llm": args.no_llm, "replay": args.replay,
                 "label_column": args.label_column, "device": args.device, "seed": args.seed,
                 "config": args.config}
    if args.model:
        overrides["llm"] = {"model": args.model}
    result = run(args.data, args.objective, args.out,
                 **{k: v for k, v in overrides.items() if v not in (None, False)})
    print(f"\nreport: {result['out_dir']}/report.html")
    print(f"verdict: {result['recommendations'].get('verdict', '')}")


if __name__ == "__main__":
    main()
