# Runtime optimization

These changes optimize Splash inference with the existing target and DFlash2
weights. They do not modify model parameters, quantization, or checkpoints.
Base-model pruning, training, recovery work, model weights, and their artifacts
are outside this branch.

All optional features are disabled by default. Target verification defaults to
eight rows. Enabling a feature does not establish a speed improvement; compare
complete generation time and output correctness on the intended workload.

## Implemented features

| Feature | Implementation | Eligibility and fallback |
| --- | --- | --- |
| Context copying with draft bypass | Match token suffixes against request history, propose the continuation, and skip the neural draft for that round. The target still verifies every proposal. Sampled verification uses an exact point-mass proposal distribution and the existing rejection/residual correction. | One unconstrained text request without a thinking budget. Matches require token equality; conflicting continuations fall back to the neural draft. History and index limits also fall back. |
| Variable target verification | Verify an active prefix of 1, 2, 4, or 8 target rows, including the anchor. The final active row supplies the bonus token. GDN folds the previous retained prefix independently of the current width; tape storage retains its maximum stride. | Narrow widths require one unconstrained dense request. Other batches and MoE targets use eight rows. The trained draft retains its eight query rows; a one-row target round needs no neural draft. |
| Adaptive verification controller | Price complete measured rounds and conditional acceptance, with context buckets and hysteresis. On M5 dense targets, strong recent wide acceptance preserves the wide block. Lookup rounds do not populate neural-draft cost estimates. | Applies to the same requests as narrow verification. Copy rounds can use eight rows independently of a pending neural probe. Other requests retain fixed eight-row verification. |
| GPU continuation between rounds | Execute two consecutive eight-row decode rounds in one submission. Device state carries the anchor, positions, accepted counts, output budget, and terminal status; GPU-patched parameters and grids prepare the next round. | One dense Qwen3.8 request with neutral greedy settings, no request flags, images, constraints, or thinking budget. Requires enough output budget and allocated KV coverage for both rounds, without crossing a KV-page or attention-split boundary. Unsupported graphs or allocation refusal fall back to a normal round. |
| Metal command replay | Cache preencoded compute indirect command buffers, refresh changing arguments, and preserve dependencies and GPU-written grids. | Requires a reusable supported command structure. Shared-event commands, dispatch profiling, and unsupported commands use ordinary encoding. See [Metal command replay](../dev/benchmarks/metal-command-replay.md) for resource lifetime, cache limits, tests, and measurements. |

GPU grouping and context copying are alternative paths for a round: a copy hit
uses ordinary target verification with the neural draft omitted. GPU grouping
requires eight target rows. Cancellation and streaming remain host-managed at
submission boundaries; a group completes at most two decode rounds before
returning control.

## Configuration

Flags are read when the native runtime is created. Restart that process after
changing them. Boolean values accept `0`, `1`, `false`, or `true`; invalid values
are rejected.

| Environment variable | Default with no overrides | Effect |
| --- | --- | --- |
| `SPLASH_PROMPT_LOOKUP` | `0` | Enable context copying and neural draft bypass. |
| `SPLASH_VERIFY_ROWS` | `8` | Select `1`, `2`, `4`, `8`, or `auto`. |
| `SPLASH_GPU_DECODE_CONTROL` | `0` | Enable eligible two-round GPU continuation. |
| `SPLASH_COMMAND_REPLAY` | `0` | Enable the bounded Metal command cache. |
| `SPLASH_RUNTIME_OPTIMIZATIONS` | `0` | Convenience switch enabling lookup, GPU continuation, command replay, and adaptive verification unless individually overridden. |

Use **fixed eight-row verification** for the M5 dense profile. Measurements of
narrow verification regressed on the evaluated workloads: smaller active
prefixes did not save enough time per round to compensate for emitting fewer
tokens. Automatic selection remains available for explicit comparisons; it is
not the recommended profile for this machine.

Set individual flags when evaluating a combination. For example, explicitly set
`SPLASH_VERIFY_ROWS=8` to retain fixed verification even when the convenience
switch is enabled. Keep other optional features off until their complete
request measurements justify enabling them.

## M5 execution limits

Active rows and hardware matrix tiles are different quantities:

- Dense quantized projections retain an **eight-row minimum matrix tile**.
  Normalization, GDN recurrence, capture, KV preparation/storage, and selection
  process the active prefix, but halving its length does not halve dense matrix
  work or weight traffic.
- Qwen3.8-27B uses six query heads per KV head. MPP requires compatible matrix
  dimensions, so its attention tile has a **four-row minimum**. One- and
  two-row verification use that tile with the appropriate active prefix.
- Draft weights and its trained eight-row query geometry are unchanged. State
  and scratch allocations retain their maximum layouts. These features do not
  reduce model weight storage or parameter count.

## Reproduce native comparisons

Run from the repository root. Supply a prepared Splash model directory and its
matching local tokenizer. The case generator needs Python with `transformers`
and reads tokenizer files locally.

```sh
SPLASH_MODEL_ROOT=/path/to/prepared/splash/model
SPLASH_TOKENIZER_ROOT="$SPLASH_MODEL_ROOT/tokenizer"
SPLASH_RESULTS_DIR=/path/to/local/runtime-results

python3 dev/tools/runtime_optimization_cases.py \
  --tokenizer "$SPLASH_TOKENIZER_ROOT" \
  --output "$SPLASH_RESULTS_DIR/cases.bin" \
  --manifest "$SPLASH_RESULTS_DIR/cases.json"

make build/engine-tests/runtime-optimization
build/engine-tests/runtime-optimization \
  build/splash.metallib "$SPLASH_MODEL_ROOT" \
  "$SPLASH_RESULTS_DIR/cases.bin" "$SPLASH_RESULTS_DIR/fixed-eight.jsonl" \
  --modes baseline,lookup,control,replay,combined \
  --trials 4 --warmup-tokens 192
```

The harness loads one target/draft pair, shares those immutable weights across
arms, and creates fresh runtime/request state for each arm. It warms every arm
before measurement and reverses arm order on alternating trials. Output is
appended to JSONL; use a fresh result filename for a separate comparison.

The native harness uses one request, INT8 KV, FP32 GDN state, and GPU prefill.
It does not measure HTTP/frontend overhead or the Neural Engine prefill split.
It checks planned allocation headroom, a 32 GiB Metal allocation limit, a
48 GiB process footprint limit, normal memory pressure, and a host availability
reserve of 15 percent. It records process memory and swap readings alongside
results.

Benchmark modes:

- `baseline`: all optional features off, eight target rows.
- `lookup`, `control`, `replay`: enable the named feature with eight target rows.
- `combined`: lookup, GPU continuation, and command replay with fixed eight rows.
- `rows1`, `rows2`, `rows4`: force the named active target width where eligible.
- `auto`: adaptive verification alone.
- `all`: enable the three optional paths and adaptive verification. This is a
  comparison arm, not the recommended M5 profile.

For an explicit width/controller comparison:

```sh
build/engine-tests/runtime-optimization \
  build/splash.metallib "$SPLASH_MODEL_ROOT" \
  "$SPLASH_RESULTS_DIR/cases.bin" "$SPLASH_RESULTS_DIR/widths.jsonl" \
  --modes baseline,rows1,rows2,rows4,auto,all \
  --trials 2 --warmup-tokens 192
```

Inspect complete `output_ids` for greedy equivalence, phase wall/GPU times,
retained/accepted counts, row distributions, lookup yield, grouped submissions,
and replay hits. Seeded sampled output can differ between speculative
strategies; distribution and rejection-correction tests cover that path.
Exclude records with `warmup: true` from measured comparisons. Report native
results separately from HTTP measurements and keep the tokenizer/case manifest
with the results.

## Measured M5 Max results

The October 24, 2026 measurements used an M5 Max with 40 GPU cores and 128 GiB
of unified memory, the unchanged `mlx-community/Qwen3.8-27B-4bit` target,
and its existing DFlash2 draft. The target snapshot was
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; the draft snapshot was
`015e795645c74b1a0eeef3b570031fb62e769bc5`.

The fixed-width comparison contains three repetitions per case and mode.
Speedups below are the median of paired latency ratios against the same
trial's flags-off control. A value above 1 favors the candidate.

| Workload | Lookup decode | Lookup total | Combined decode | Combined total |
| --- | ---: | ---: | ---: | ---: |
| Copy a module | 1.032x | 1.033x | 1.022x | 1.018x |
| Rename a function in a module | 1.071x | 1.057x | 1.059x | 1.041x |
| Write a new function | 0.988x | 0.977x | 0.984x | 0.984x |
| Short arithmetic answer | 1.008x | 1.022x | 1.026x | 1.044x |
| Retrieve from a 6,602-token prompt | 1.005x | 1.004x | 0.992x | 1.006x |
| Sampled module copy | 1.094x | 1.068x | 1.056x | 1.040x |

These are small synthetic inference fixtures, not HumanEval, MBPP, or a general
agent benchmark. Three repetitions do not establish statistical significance;
small differences can be timing noise. The arithmetic response is incorrect
in the original and every candidate (`889`, expected `869`), so that row is a
latency comparison of a failed task.

All **30 candidate greedy outputs** in this comparison match both its local
control and the unmodified `64954ec` build. All nine sampled-copy task checks
pass. The original build was measured in a separate process/run; its timings
are preserved for inspection, not used to attribute the paired improvements.

The separate 108-measurement matrix exercises each feature individually.
Automatic rows reduced new-function decode throughput to **0.647x** its paired
control. GPU continuation and command replay execute successfully, but neither
establishes a broad model-level speed gain here. Keep them opt-in. The measured
profile for copy/edit workloads is shown below. Use an existing prepared model
directory and a Python environment with Splash's server dependencies installed.
This starts the built server directly and reuses the supplied weights.

```sh
SPLASH_MODEL_ROOT=/path/to/prepared/splash/model
SPLASH_PROMPT_LOOKUP=1 SPLASH_VERIFY_ROWS=8 \
SPLASH_GPU_DECODE_CONTROL=0 SPLASH_COMMAND_REPLAY=0 \
  python3 -m server.server "$SPLASH_MODEL_ROOT" \
  --model mlx-community/Qwen3.8-27B-4bit \
  --tokenizer "$SPLASH_MODEL_ROOT/tokenizer" --binary build/splash \
  --host 127.0.0.1 --port 8000 --max-memory 32G --max-context 32768
```

The fixed-width native run peaked at **17.74 GiB Metal allocation** and
**18.26 GiB process footprint**. The independent watchdog observed at least
**86.83 GiB available memory**, with no swap increase or guard abort.
Weights and the maximum state/scratch layouts remain the same size.

[Raw measurements and reproduction notes](../dev/benchmarks/results/m5-max-runtime-2026-10-24/README.md)
include the case manifest, all output token IDs, feature counters, phase times,
and memory observations.

## Production HTTP validation

The production server also passed its complete real-model HTTP smoke on
October 24, 2026, with lookup, GPU continuation, and command replay enabled,
fixed eight-row verification, a **32 GiB Metal budget**, a **32,768-token
context limit**, and the disk cache disabled. It served the same unchanged
target/draft installation, including its vision components.

The run covered chat and SSE streaming, tool calls, strict JSON schemas,
thinking budgets under grammar constraints, sampling penalties and filters,
Responses and Messages compatibility, image/PDF input and prefix reuse,
scoring alongside generation, and scoring timeout recovery. A separate long
output check delivered **128 completion tokens** through both ordinary and
streaming responses, with identical greedy text and usage; the stream
contained 131 SSE chunks and its final completion marker.

There were **93 submitted requests: 92 completed and one expected scoring
timeout**, followed by successful recovery. Transport restarts, Metal failures,
and capacity failures were all zero. The run recorded 2,821 generated tokens
and 31,896 prefill input tokens. Native protocol regression tests separately
verify that a 17-token grouped result is split into the existing nine-token
wire frames, with grouped speculative cycles counted individually.

The guarded run took 117.4 seconds including startup. The device reported a
**22.97 GiB peak Metal allocation**; the final governor charge was 23.71 GiB.
The watchdog observed at least 80.03 GiB host availability and normal memory
pressure throughout. System swap remained 136,445,952 bytes before and after,
and both owned server processes exited. Process RSS is recorded separately
and does not represent total Metal/unified-memory consumption.

Reproduce the standard HTTP smoke after building the native executable:

```sh
SPLASH_PROMPT_LOOKUP=1 SPLASH_GPU_DECODE_CONTROL=1 \
SPLASH_COMMAND_REPLAY=1 SPLASH_VERIFY_ROWS=8 \
  python3 dev/tests/smoke_real.py \
  --model mlx-community/Qwen3.8-27B-4bit \
  --model-root "$SPLASH_MODEL_ROOT" --binary build/splash \
  --max-memory 32G --max-context 32768 --max-cache-disk 0 --kv-format int8
```

The [portable HTTP summary](../dev/benchmarks/results/m5-max-runtime-2026-10-24/http-validation.json)
retains the configuration, checks, counters, and memory observations. This
validates production behavior; it does not establish an HTTP speed improvement.

## Final validation

The complete `make test-engine-cpu` and `make test-engine-metal` suites passed.
The latter runs shader bounds validation and includes the new GPU control,
variable-row, dependency, and resource-lifetime tests. Its Neural Engine fault
injection intentionally produces timeout/failure warnings and verifies recovery.
The isolated replay suite also passed with both API and shader validation.
All ten Python report unit tests passed.

The existing full real-model oracle passed with the default eight-row path,
INT8 KV, FP32 states, and GPU prefill. It covers B1–B4, masks, sampling,
thinking budgets, images, state captures, preemption, and cache restoration.
It ran for 162.2 seconds with no watchdog abort or swap increase.

[Validation commands and log hashes](../dev/benchmarks/results/m5-max-runtime-2026-10-24/validation.json)
and the [complete oracle output](../dev/benchmarks/results/m5-max-runtime-2026-10-24/runtime-oracle.log)
are retained with the measurements.
