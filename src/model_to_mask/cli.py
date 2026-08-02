from __future__ import annotations

import argparse
from pathlib import Path

from .config import CompilerConfig
from .pipeline import initialize_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize a model-to-mask compiler workspace."
    )
    parser.add_argument("--model", required=True, help="Path to the quantized PyTorch model")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where compiler artifacts and configs will be created",
    )
    parser.add_argument("--top-name", required=True, help="Top-level hardware module name")
    parser.add_argument(
        "--output-format",
        choices=("rtlil", "blif"),
        default="rtlil",
        help="Structural export format for the backend handoff",
    )
    parser.add_argument(
        "--weight-split-ratio",
        type=float,
        default=0.95,
        help="Fraction of weights mapped to mask ROM in the hybrid architecture",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = CompilerConfig(
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        top_name=args.top_name,
        output_format=args.output_format,
        weight_split_ratio=args.weight_split_ratio,
    )
    manifest = initialize_workspace(config)
    print(f"Initialized model-to-mask workspace: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
