# model-to-mask

m2m is an early scaffold for a model-to-mask compiler pipeline targeting fixed-weight
neural ASIC flows.

## Current status

This repository now contains a minimal Python project that bootstraps the compiler
workspace for:

- frontend ingestion from quantized PyTorch into TOSA MLIR
- TOSA to linalg and linalg bufferization checkpoints
- CIRCT scheduling and structural lowering placeholders
- Yosys synthesis handoff configuration
- OpenROAD physical design handoff configuration

The current implementation does **not** perform the full lowering flow yet. It
creates a reproducible workspace layout, stage manifest, and starter config files so
the real compiler integrations can be added incrementally.

## Usage

From `/home/runner/work/model-to-mask/model-to-mask`:

```bash
python -m model_to_mask.cli \
  --model examples/quantized_model.pt \
  --output-dir build/demo \
  --top-name demo_accelerator \
  --print-plan
```

You can also initialize from a JSON config file:

```bash
python -m model_to_mask.cli --config examples/compiler-config.json
```

Or continue from an existing TOSA checkpoint instead of generating a placeholder:

```bash
python -m model_to_mask.cli \
  --model examples/quantized_model.pt \
  --output-dir build/demo \
  --top-name demo_accelerator \
  --tosa-input /absolute/path/to/model.tosa.mlir
```

This creates:

- `build/demo/manifest.json`
- `build/demo/mlir/`
- `build/demo/circt/`
- `build/demo/backend/`
- `build/demo/config/`
- `build/demo/reports/`

Generated config assets include:

- `config/compiler-config.json`
- `config/toolchain.json`
- `backend/backend.env`
- `reports/stage-status.json`
- `reports/command-plan.json`
- `reports/repository-state.json`

The printed command plan captures the intended handoff sequence for:

- torch-mlir ingestion into TOSA
- TOSA to linalg and linalg bufferization
- CIRCT scheduling and structural lowering
- Yosys backend export
- OpenROAD physical design execution

When `--tosa-input` is provided, the frontend stage switches from a planned
torch-mlir ingestion step to importing that existing TOSA MLIR artifact so the
rest of the workspace can continue from the checkpointed frontend output.

The generated manifest and repository state report also record which local checkout
produced the workspace, including the repository root, current branch, commit SHA,
and whether the checkout had uncommitted changes.

To run the test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
