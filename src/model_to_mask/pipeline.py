from __future__ import annotations

import json
from pathlib import Path

from .config import CompilerConfig


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def initialize_workspace(config: CompilerConfig) -> Path:
    for directory in config.stage_paths().values():
        directory.mkdir(parents=True, exist_ok=True)

    directories = config.stage_paths()
    artifacts = config.artifact_paths()

    _write_text(
        artifacts["tosa_mlir"],
        "\n".join(
            [
                "// Placeholder TOSA MLIR artifact",
                f"// source_model={config.model_path}",
                "// TODO: import quantized PyTorch with torch-mlir into TOSA.",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["linalg_mlir"],
        "\n".join(
            [
                "// Placeholder linalg MLIR artifact",
                "// pass_pipeline: -tosa-to-linalg",
                "",
            ]
        ),
    )
    _write_text(
        artifacts["bufferized_mlir"],
        "\n".join(
            [
                "// Placeholder bufferized MLIR artifact",
                "// pass_pipeline: -linalg-bufferize",
                "",
            ]
        ),
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
                    "status": "planned",
                    "output": str(artifacts["tosa_mlir"]),
                },
                "mlir_lowering": {
                    "status": "planned",
                    "output": str(artifacts["linalg_mlir"]),
                },
                "bufferization": {
                    "status": "planned",
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

    artifacts["manifest"].write_text(
        json.dumps(config.to_manifest(), indent=2) + "\n",
        encoding="utf-8",
    )
    return artifacts["manifest"]
