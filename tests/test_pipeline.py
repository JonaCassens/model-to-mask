from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from model_to_mask.cli import build_parser, main
from model_to_mask.config import CompilerConfig
from model_to_mask.pipeline import build_command_plan, initialize_workspace


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

    def test_loads_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "examples/model.pt",
                        "output_dir": "build/from-file",
                        "top_name": "file_top",
                        "output_format": "blif",
                        "weight_split_ratio": 0.9,
                        "toolchain": {"yosys": "custom-yosys"},
                    }
                ),
                encoding="utf-8",
            )

            config = CompilerConfig.from_file(config_path)

            self.assertEqual(config.top_name, "file_top")
            self.assertEqual(config.output_format, "blif")
            self.assertEqual(config.weight_split_ratio, 0.9)
            self.assertEqual(config.toolchain.yosys, "custom-yosys")


class PipelineInitializationTests(unittest.TestCase):
    def test_command_plan_tracks_expected_stages(self) -> None:
        config = CompilerConfig(
            model_path=Path("examples/model.pt"),
            output_dir=Path("build/demo"),
            top_name="demo_top",
        )

        plan = build_command_plan(config)

        self.assertEqual(
            [step["stage"] for step in plan],
            [
                "frontend_ingestion",
                "mlir_lowering",
                "bufferization",
                "circt_scheduling",
                "structural_lowering",
                "backend_export",
                "physical_design",
            ],
        )
        self.assertEqual(plan[0]["tool"], "torch-mlir-opt")
        self.assertIn("--emit-tosa", plan[0]["command"])

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
            self.assertTrue((output_dir / "config" / "compiler-config.json").exists())
            self.assertTrue((output_dir / "config" / "toolchain.json").exists())
            self.assertTrue((output_dir / "backend" / "backend.env").exists())
            self.assertTrue((output_dir / "reports" / "stage-status.json").exists())
            self.assertTrue((output_dir / "reports" / "command-plan.json").exists())

    def test_cli_parser_exposes_expected_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--model", "model.pt", "--output-dir", "build/out", "--top-name", "top"]
        )
        self.assertIsNone(args.output_format)
        self.assertIsNone(args.weight_split_ratio)

    def test_cli_can_initialize_from_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "compiler.json"
            output_dir = Path(tmpdir) / "workspace"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "examples/model.pt",
                        "output_dir": str(output_dir),
                        "top_name": "cfg_top",
                    }
                ),
                encoding="utf-8",
            )

            import sys

            old_argv = sys.argv
            sys.argv = [
                "model_to_mask.cli",
                "--config",
                str(config_path),
            ]
            try:
                exit_code = main()
            finally:
                sys.argv = old_argv

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "manifest.json").exists())

    def test_cli_print_plan_outputs_stage_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "workspace"
            import sys

            old_argv = sys.argv
            sys.argv = [
                "model_to_mask.cli",
                "--model",
                "examples/model.pt",
                "--output-dir",
                str(output_dir),
                "--top-name",
                "stdout_top",
                "--print-plan",
            ]
            buffer = StringIO()
            try:
                with redirect_stdout(buffer):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            self.assertEqual(exit_code, 0)
            output = buffer.getvalue()
            self.assertIn('"stage": "frontend_ingestion"', output)
            self.assertIn("Initialized model-to-mask workspace:", output)


if __name__ == "__main__":
    unittest.main()
