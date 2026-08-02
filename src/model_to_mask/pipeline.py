from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import CompilerConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_artifact(source: Path, output_path: Path) -> None:
    _write_text(output_path, source.read_text(encoding="utf-8"))


def collect_repository_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "available": False,
            "root": str(REPOSITORY_ROOT),
            "branch": None,
            "commit": None,
            "is_dirty": None,
        }

    return {
        "available": True,
        "root": str(REPOSITORY_ROOT),
        "branch": branch,
        "commit": commit,
        "is_dirty": bool(dirty_output),
    }


def _seeded_stage_plan(
    *,
    stage: str,
    source: Path | None,
    output: Path,
    planned_tool: str,
    planned_inputs: list[str],
    planned_command: list[str],
) -> dict[str, object]:
    if source is not None:
        return {
            "stage": stage,
            "tool": "copy",
            "inputs": [str(source)],
            "outputs": [str(output)],
            "command": [
                "cp",
                str(source),
                str(output),
            ],
        }
    return {
        "stage": stage,
        "tool": planned_tool,
        "inputs": planned_inputs,
        "outputs": [str(output)],
        "command": planned_command,
    }


def _initialize_seeded_artifact(
    source: Path | None,
    output_path: Path,
    placeholder_lines: list[str],
) -> None:
    if source is not None:
        _copy_artifact(source, output_path)
        return
    _write_text(
        output_path,
        "\n".join(placeholder_lines + [""]),
    )


def build_command_plan(config: CompilerConfig) -> list[dict[str, object]]:
    artifacts = config.artifact_paths()
    return [
        _seeded_stage_plan(
            stage="frontend_ingestion",
            source=config.tosa_input_path,
            output=artifacts["tosa_mlir"],
            planned_tool=config.toolchain.torch_mlir_opt,
            planned_inputs=[str(config.model_path)],
            planned_command=[
                config.toolchain.torch_mlir_opt,
                str(config.model_path),
                "--emit-tosa",
                "-o",
                str(artifacts["tosa_mlir"]),
            ],
        ),
        _seeded_stage_plan(
            stage="mlir_lowering",
            source=config.linalg_input_path,
            output=artifacts["linalg_mlir"],
            planned_tool=config.toolchain.torch_mlir_opt,
            planned_inputs=[str(artifacts["tosa_mlir"])],
            planned_command=[
                config.toolchain.torch_mlir_opt,
                str(artifacts["tosa_mlir"]),
                "-tosa-to-linalg",
                "-o",
                str(artifacts["linalg_mlir"]),
            ],
        ),
        _seeded_stage_plan(
            stage="bufferization",
            source=config.bufferized_input_path,
            output=artifacts["bufferized_mlir"],
            planned_tool=config.toolchain.torch_mlir_opt,
            planned_inputs=[str(artifacts["linalg_mlir"])],
            planned_command=[
                config.toolchain.torch_mlir_opt,
                str(artifacts["linalg_mlir"]),
                "-linalg-bufferize",
                "-o",
                str(artifacts["bufferized_mlir"]),
            ],
        ),
        {
            "stage": "circt_scheduling",
            "tool": config.toolchain.circt_opt,
            "inputs": [str(artifacts["bufferized_mlir"])],
            "outputs": [
                str(artifacts["calyx_mlir"]),
                str(artifacts["handshake_mlir"]),
            ],
            "command": [
                config.toolchain.circt_opt,
                str(artifacts["bufferized_mlir"]),
                "--lower-to-calyx-and-handshake",
            ],
        },
        {
            "stage": "structural_lowering",
            "tool": config.toolchain.firtool,
            "inputs": [
                str(artifacts["calyx_mlir"]),
                str(artifacts["handshake_mlir"]),
            ],
            "outputs": [str(artifacts["structural_mlir"])],
            "command": [
                config.toolchain.firtool,
                str(artifacts["calyx_mlir"]),
                "--merge-handshake",
                str(artifacts["handshake_mlir"]),
                "-o",
                str(artifacts["structural_mlir"]),
            ],
        },
        {
            "stage": "backend_export",
            "tool": config.toolchain.yosys,
            "inputs": [
                str(artifacts["structural_mlir"]),
                str(artifacts["yosys_script"]),
            ],
            "outputs": [str(artifacts["netlist"])],
            "command": [
                config.toolchain.yosys,
                "-c",
                str(artifacts["yosys_script"]),
            ],
        },
        {
            "stage": "physical_design",
            "tool": config.toolchain.openroad,
            "inputs": [
                str(artifacts["netlist"]),
                str(artifacts["openroad_script"]),
            ],
            "outputs": [str(directories := config.stage_paths()["reports"] / "openroad-final.def")],
            "command": [
                config.toolchain.openroad,
                str(artifacts["openroad_script"]),
            ],
        },
    ]


def initialize_workspace(config: CompilerConfig) -> Path:
    for directory in config.stage_paths().values():
        directory.mkdir(parents=True, exist_ok=True)

    directories = config.stage_paths()
    artifacts = config.artifact_paths()
    repository_state = collect_repository_state()

    _initialize_seeded_artifact(
        config.tosa_input_path,
        artifacts["tosa_mlir"],
        [
            "// Placeholder TOSA MLIR artifact",
            f"// source_model={config.model_path}",
            "// TODO: import quantized PyTorch with torch-mlir into TOSA.",
        ],
    )
    _initialize_seeded_artifact(
        config.linalg_input_path,
        artifacts["linalg_mlir"],
        [
            "// Placeholder linalg MLIR artifact",
            "// pass_pipeline: -tosa-to-linalg",
        ],
    )
    _initialize_seeded_artifact(
        config.bufferized_input_path,
        artifacts["bufferized_mlir"],
        [
            "// Placeholder bufferized MLIR artifact",
            "// pass_pipeline: -linalg-bufferize",
        ],
    )
    _write_text(
        artifacts["calyx_mlir"],
        "\n".join(
            [
                "// Placeholder CIRCT calyx schedule",
                "// TODO: lower static schedules to calyx.",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["handshake_mlir"],
        "\n".join(
            [
                "// Placeholder CIRCT handshake schedule",
                "// TODO: lower dynamic dataflow segments to handshake.",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["structural_mlir"],
        "\n".join(
            [
                "// Placeholder CIRCT structural hw/comb artifact",
                "// TODO: lower scheduled IR into hw and comb.",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["weight_map"],
        json.dumps(
            {
                "strategy": "95/5-hybrid",
                "frozen_weight_ratio": config.weight_split_ratio,
                "frozen_weights_target": "1cpp_mask_rom_blackbox",
                "adaptation_target": "hw.module.sram",
            },
            indent=2,
        )
        + "\n",
    )
    _write_text(
        artifacts["netlist"],
        "\n".join(
            [
                f"# Placeholder {config.output_format.upper()} export",
                "# TODO: export structural CIRCT output without generating huge Verilog.",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["yosys_script"],
        "\n".join(
            [
                f"read_{config.output_format} {artifacts['netlist'].name}",
                "# TODO: set liberty path",
                "hierarchy -check -top ${TOP}",
                "proc; opt",
                "techmap; opt",
                "abc -liberty ${LIBERTY}",
                "stat",
                "write_json mapped_netlist.json",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["openroad_script"],
        "\n".join(
            [
                "# TODO: set technology, LEF/DEF/liberty, and floorplan inputs",
                "read_verilog mapped_netlist.v",
                "link_design $::env(TOP)",
                "# Placeholder placement directives for RePlAce / TritonCTS / TritonRoute",
                "",
            ]
        ),
    )
    _write_text(
        directories["config"] / "pass-pipelines.txt",
        "\n".join(
            [
                "frontend_ingestion: torch-mlir -> tosa",
                "mlir_lowering: -tosa-to-linalg",
                "bufferization: -linalg-bufferize",
                "scheduling: calyx | handshake",
                "structural_lowering: hw + comb",
                f"export: {config.output_format}",
                "",
            ]
        ),
    )
    _write_text(
        directories["config"] / "compiler-config.json",
        json.dumps(config.to_manifest(), indent=2) + "\n",
    )
    _write_text(
        directories["config"] / "toolchain.json",
        json.dumps(config.to_manifest()["toolchain"], indent=2) + "\n",
    )
    _write_text(
        directories["backend"] / "backend.env",
        "\n".join(
            [
                f"TOP={config.top_name}",
                "LIBERTY=path/to/standard_cells.lib",
                "TECH_LEF=path/to/tech.lef",
                "SC_LEF=path/to/standard_cells.lef",
                "",
            ]
        ),
    )
    _write_text(
        directories["reports"] / "stage-status.json",
        json.dumps(
            {
                "frontend_ingestion": {
                    "status": "imported" if config.tosa_input_path is not None else "planned",
                    "source": (
                        str(config.tosa_input_path)
                        if config.tosa_input_path is not None
                        else str(config.model_path)
                    ),
                    "output": str(artifacts["tosa_mlir"]),
                },
                "mlir_lowering": {
                    "status": "imported" if config.linalg_input_path is not None else "planned",
                    "source": (
                        str(config.linalg_input_path)
                        if config.linalg_input_path is not None
                        else str(artifacts["tosa_mlir"])
                    ),
                    "output": str(artifacts["linalg_mlir"]),
                },
                "bufferization": {
                    "status": (
                        "imported" if config.bufferized_input_path is not None else "planned"
                    ),
                    "source": (
                        str(config.bufferized_input_path)
                        if config.bufferized_input_path is not None
                        else str(artifacts["linalg_mlir"])
                    ),
                    "output": str(artifacts["bufferized_mlir"]),
                },
                "circt_scheduling": {
                    "status": "planned",
                    "outputs": [
                        str(artifacts["calyx_mlir"]),
                        str(artifacts["handshake_mlir"]),
                    ],
                },
                "structural_lowering": {
                    "status": "planned",
                    "output": str(artifacts["structural_mlir"]),
                },
                "backend_export": {
                    "status": "planned",
                    "output": str(artifacts["netlist"]),
                },
                "physical_design": {
                    "status": "planned",
                    "output": str(artifacts["openroad_script"]),
                },
            },
            indent=2,
        )
        + "\n",
    )
    _write_text(
        directories["reports"] / "command-plan.json",
        json.dumps(build_command_plan(config), indent=2) + "\n",
    )
    _write_text(
        directories["reports"] / "repository-state.json",
        json.dumps(repository_state, indent=2) + "\n",
    )

    manifest = config.to_manifest()
    manifest["repository_state"] = repository_state
    artifacts["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifacts["manifest"]
