from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CompilerConfig
from .pipeline import build_command_plan, initialize_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize a model-to-mask compiler workspace."
    )
    parser.add_argument(
        "--config",
        help="Optional path to a JSON compiler config file; CLI flags override file values",
    )
    parser.add_argument("--model", help="Path to the quantized PyTorch model")
    parser.add_argument(
        "--output-dir",
        help="Directory where compiler artifacts and configs will be created",
    )
    parser.add_argument("--top-name", help="Top-level hardware module name")
    parser.add_argument(
        "--output-format",
        choices=("rtlil", "blif"),
        help="Structural export format for the backend handoff",
    )
    parser.add_argument(
        "--weight-split-ratio",
        type=float,
        help="Fraction of weights mapped to mask ROM in the hybrid architecture",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the planned stage commands after workspace initialization",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.config:
        config = CompilerConfig.from_file(
            args.config,
            model_path=Path(args.model) if args.model else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            top_name=args.top_name,
            output_format=args.output_format,
            weight_split_ratio=args.weight_split_ratio,
        )
    else:
        missing = [
            name
            for name, value in {
                "--model": args.model,
                "--output-dir": args.output_dir,
                "--top-name": args.top_name,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"missing required arguments without --config: {', '.join(missing)}")
        config = CompilerConfig(
            model_path=Path(args.model),
            output_dir=Path(args.output_dir),
            top_name=args.top_name,
            output_format=args.output_format or "rtlil",
            weight_split_ratio=args.weight_split_ratio or 0.95,
        )
    manifest = initialize_workspace(config)
    if args.print_plan:
        print(json.dumps(build_command_plan(config), indent=2))
    print(f"Initialized model-to-mask workspace: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
