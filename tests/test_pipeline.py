from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_to_mask.cli import build_parser
from model_to_mask.config import CompilerConfig
from model_to_mask.pipeline import initialize_workspace


class CompilerConfigTests(unittest.TestCase):
    def test_rejects_unknown_output_format(self) -> None:
        with self.assertRaises(ValueError):
            CompilerConfig(
                model_path=Path("model.pt"),
                output_dir=Path("build/demo"),
                top_name="demo",
                output_format="verilog",
            )

    def test_rejects_invalid_weight_split_ratio(self) -> None:
        with self.assertRaises(ValueError):
            CompilerConfig(
                model_path=Path("model.pt"),
                output_dir=Path("build/demo"),
                top_name="demo",
                weight_split_ratio=1.2,
            )


class PipelineInitializationTests(unittest.TestCase):
    def test_workspace_initialization_creates_manifest_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "build"
            config = CompilerConfig(
                model_path=Path("examples/model.pt"),
                output_dir=output_dir,
                top_name="demo_top",
                output_format="blif",
            )

            manifest_path = initialize_workspace(config)

            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["top_name"], "demo_top")
            self.assertEqual(manifest["output_format"], "blif")

            for artifact in manifest["artifacts"].values():
                self.assertTrue(Path(artifact).exists())

    def test_cli_parser_exposes_expected_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--model", "model.pt", "--output-dir", "build/out", "--top-name", "top"]
        )
        self.assertEqual(args.output_format, "rtlil")
        self.assertEqual(args.weight_split_ratio, 0.95)


if __name__ == "__main__":
    unittest.main()
