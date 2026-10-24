# M5 Max runtime measurements — October 24, 2026

These are inference measurements with unchanged model weights. See the
[runtime report](../../../../docs/runtime-optimization.md#measured-m5-max-results)
for interpretation, hardware, eligibility, and recommended flags.

## Files

- `cases.json`: the six English prompts, sampling settings, token fingerprints,
  and SHA-256 of the generated binary case file. The tokenizer path is portable.
- `baseline-original.jsonl`: 18 measured cases plus warmup, linked against the
  unmodified `64954ec` runtime and its original shaders.
- `revised-matrix.jsonl`: 108 measured cases plus six warmups, comparing the
  control, lookup, automatic rows, GPU continuation, replay, and all features.
- `fixed-width-matrix.jsonl`: 54 measured cases plus three warmups, comparing
  the control, lookup, and combined lookup/continuation/replay at eight rows.
- `resources.json`: independent watchdog summaries for the guarded native runs.
- `validation.json`: final passing CPU/Metal suite commands, replay API/shader
  validation, report unit tests, and hashes of the external full suite logs.
- `runtime-oracle.log`: the complete existing real-model oracle with optional
  runtime paths disabled, INT8 KV, FP32 states, and GPU prefill. This checks
  default-path compatibility, B1–B4, constraints, sampling, images, state
  captures, preemption, and cache restore. It passed without a watchdog abort
  or swap increase; its resource summary is also in `resources.json`.
- `http-validation.json`: condensed production HTTP smoke, streaming and
  resource observations with lookup, continuation, and replay enabled at
  eight rows. The full server log and per-second watchdog log remain local;
  no model paths or machine-specific filesystem paths are included here.

Each run loads one target and draft. No two model processes ran concurrently.
Watchdogs stopped a run if available memory fell below 16 GiB, process RSS
exceeded 48 GiB, swap increased by more than 256 MiB, or elapsed time exceeded
30 minutes. The native harness additionally checks Metal allocation, physical
footprint, host reserve, and memory pressure. None of these runs aborted.

## Reproduce the report

Run from the repository root with the same local tokenizer used for the case
generator. The tokenizer is external; no model/tokenizer weights are included.

```sh
SPLASH_RESULTS_DIR=dev/benchmarks/results/m5-max-runtime-2026-10-24
SPLASH_TOKENIZER_ROOT=/path/to/prepared/model/tokenizer

python3 dev/tools/runtime_optimization_report.py \
  "$SPLASH_RESULTS_DIR/fixed-width-matrix.jsonl" \
  --cases "$SPLASH_RESULTS_DIR/cases.json" \
  --tokenizer "$SPLASH_TOKENIZER_ROOT" \
  --reference "$SPLASH_RESULTS_DIR/baseline-original.jsonl" \
  --json-output /tmp/splash-runtime-report.json \
  --markdown-output /tmp/splash-runtime-report.md
```

Warmups are excluded. Exact token equality applies to greedy generations;
sampled copies are checked against the requested source. The arithmetic fixture
fails in the original model too and remains a failed task in the report.

The measured harness predates optional prompt fingerprints in each output row.
All three runs used the same immutable `cases.bin`, whose checksum and exact
prompts are retained in `cases.json`. Do not pair these rows with newly changed
prompts just because case names and token counts match. Recreate `cases.bin`
using the documented generator and verify its checksum before a new comparison.
