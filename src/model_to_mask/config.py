from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ToolchainConfig:
    torch_mlir_opt: str = "torch-mlir-opt"
    circt_opt: str = "circt-opt"
    firtool: str = "firtool"
    yosys: str = "yosys"
    openroad: str = "openroad"


@dataclass(slots=True)
class CompilerConfig:
    model_path: Path
    output_dir: Path
    top_name: str
    output_format: str = "rtlil"
    weight_split_ratio: float = 0.95
    toolchain: ToolchainConfig = ToolchainConfig()

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.output_dir = Path(self.output_dir)
        self.output_format = self.output_format.lower()
        if self.output_format not in {"rtlil", "blif"}:
            raise ValueError("output_format must be either 'rtlil' or 'blif'")
        if not 0 < self.weight_split_ratio < 1:
            raise ValueError("weight_split_ratio must be between 0 and 1")

    def stage_paths(self) -> dict[str, Path]:
        return {
            "config": self.output_dir / "config",
            "mlir": self.output_dir / "mlir",
            "circt": self.output_dir / "circt",
            "backend": self.output_dir / "backend",
            "reports": self.output_dir / "reports",
        }

    def artifact_paths(self) -> dict[str, Path]:
        return {
            "tosa_mlir": self.output_dir / "mlir" / "model.tosa.mlir",
            "linalg_mlir": self.output_dir / "mlir" / "model.linalg.mlir",
            "bufferized_mlir": self.output_dir / "mlir" / "model.bufferized.mlir",
            "calyx_mlir": self.output_dir / "circt" / "model.calyx.mlir",
            "handshake_mlir": self.output_dir / "circt" / "model.handshake.mlir",
            "structural_mlir": self.output_dir / "circt" / "model.hw.mlir",
            "weight_map": self.output_dir / "circt" / "weight_partition.json",
            "netlist": self.output_dir / "backend" / f"model.{self.output_format}",
            "yosys_script": self.output_dir / "backend" / "synth.ys",
            "openroad_script": self.output_dir / "backend" / "flow.tcl",
            "manifest": self.output_dir / "manifest.json",
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            "top_name": self.top_name,
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir),
            "output_format": self.output_format,
            "weight_split_ratio": self.weight_split_ratio,
            "toolchain": asdict(self.toolchain),
            "directories": {key: str(value) for key, value in self.stage_paths().items()},
            "artifacts": {key: str(value) for key, value in self.artifact_paths().items()},
        }
