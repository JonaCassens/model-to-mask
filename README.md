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
  --top-name demo_accelerator
```

You can also initialize from a JSON config file:

```bash
python -m model_to_mask.cli --config compiler-config.json
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

To run the test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
