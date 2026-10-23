# Native DFlash2 adaptation for the pruned Qwen target

## Scope and sources

This branch ports the completed-buffer capture idea from Splash
[#550](https://github.com/incoai/splash/pull/550) (`ef0e95d4`) and the final
hidden readback from [#556](https://github.com/incoai/splash/pull/556)
(`b4006b12`). It adds a small MLX selector-projection fit against the actual
deployed Q4 target. It does not merge the experimental training PR stack.

The open [MPS trainer #537](https://github.com/incoai/splash/pull/537) is a
useful Apple training foundation. Its family validation and BF16 teacher
capture do not directly accept this 52-layer Q4 target. The
[#554 recall scopes](https://github.com/incoai/splash/pull/554) are the next
reference if the correct token is missing from the natural candidate set.
The research selectors in [#553](https://github.com/incoai/splash/pull/553)
require additional native serving support.

## Components

- `draft_capture.mm`: one greedy native request, existing arena readback,
  bounded context/generation, 24 GiB planned allocation guard and memory
  telemetry. An instrumented `Runtime.cpp` object supplies the accessors;
  the ordinary runtime object omits them. No shader or arena changes.
- `capture_dflash_suite.py`: tokenizer-only prompt preparation and sequential
  child processes with a macOS memory-pressure check before each load.
- `analyze_dflash_capture.py`: candidate recall, selector mistakes, censored
  prefix labels, native score replay and conditional training references.
- `fit_dflash_selector.py`: FP32 Adam masters for only
  `candidate_selector.hidden_projection.weight`, 1,310,720 parameters.
  Codebooks and backbone remain frozen and are never loaded as a full model.

The capture CLI accepts already-tokenized little-endian uint32 prompts.
The suite prepares its prompts with thinking disabled. Hidden rows are raw
BF16 **bit patterns**, not numeric uint16 activations. Row zero is the anchor;
rows 1–7 align with candidate rows 0–6.

Only accepted positions and an unambiguously verified first rejection provide
labels. All later verifier rows are censored. Missing gold tokens count as
recall failures and are excluded from conditional CE; they are never inserted
into top-16. Prompt-level splits prevent positions from the same request
appearing in both training and validation.

New captures include the SHA256 of their exact little-endian prompt token
bytes; the suite checks it against its manifest. The fitter rejects duplicate
prompt hashes, checks token files and binds `--draft` to the captured assembly's
draft (same files, or identical streamed hashes). The analyzer requires a
terminal summary, so an interrupted capture cannot publish successful results.
The initial pilot captures predate the prompt hash field; their original
commands and distinct token files were retained and checked separately.

The fitter reproduces the loader's BF16 source rounding, affine 4-bit groups
of 64, BF16 scales/biases, FP32 reconstruction and BF16 projection output.
It checks score/argmax parity before training. Validation conditional CE selects
the best checkpoint, with the unmodified original as a fallback. Export
streams a separate checkpoint and replaces only the projection's same-size
BF16 byte range. The original target and draft files are not overwritten.

## Local experiment, 2026-10-23

Artifacts are in
`~/Models/Splash/Qwen3.8-pruned/draft-adaptation-20261023/`.
The target is the 19.876B logical-parameter, 52-layer, MLP-14,336 Q4 candidate,
with taps `[5,15,25,39,49]`. Eight authored coding/tool contexts contain
95–810 prompt tokens, thinking disabled, temperature zero, int8 KV and FP32
recurrent state. Four prompts train and four validate. This is a diagnostic
corpus, not an independent coding-quality benchmark.

Initial capture limits were 48 cycles / 256 generated tokens. Some long
responses therefore stop at the cycle limit; do not compare these timing
numbers as equal-work throughput measurements.

| Initial capture | Full 64-layer Q4 target | Pruned Q4 target |
| --- | ---: | ---: |
| Native cycles | 223 | 228 |
| Trusted token labels | 882 | 791 |
| Natural top-16 recall at reached trusted positions | 97.51% | 94.44% |
| Verified rejections | 165 | 195 |
| Rejections with gold present in natural top-16 | 143 | 151 |
| Peak Metal allocated memory | 16.74 GiB | 12.94 GiB |

All 1,596 proposed choices in the pruned capture replayed exactly. Its 151
covered rejections represent 77.44% of observed rejections, so projection
adaptation has measurable local headroom. That fraction is not a promised
recoverable speedup: changed proposals alter later draft context and state.
The base and pruned targets also generate different continuations, so their
recall rows are observations on their own reached prefixes.

The first fit used 100 steps, batch 16, learning rate 1e-5 and seed zero.
The best validation CE occurred at step 80:

| Cached validation metric | Released projection | Selected projection |
| --- | ---: | ---: |
| Conditional CE | 0.637116 | 0.628893 |
| Conditional accuracy, 335 naturally covered rows | 80.90% | 81.79% |

Reconstructed baseline argmax matched all 791 trusted native rows. Mean
absolute edge-score error was about 2e-5. Training peaked at 98.93 MiB in MLX
and 362 MiB process RSS. Streaming byte comparison verified that all bytes
outside the 2,621,440-byte projection range were identical in the exported
3.85 GB draft checkpoint.

The first native rerun retained identical output-token prefixes on all eight
prompts. Four validation prompts showed a small raw acceptance change from
40.77% to 41.24%; the captures stopped at different continuation lengths in
some cases. A separate matched-token ABBA comparison is recorded in
`native-abba/report.json`. Every round produced the same 515 output tokens
across the four validation prompts, with identical token hashes:

| Matched native validation | Released projection | Selected projection |
| --- | ---: | ---: |
| Cycles per round | 159 | 155 |
| Raw accepted/offered draft positions | 32.08% | 33.27% |
| Median decode GPU tokens/s | 109.13 | 114.07 |
| Within-model round spread | 4.93% | 0.14% |

The deterministic cycle reduction is 2.52%. The observed median GPU throughput
increase is 4.53%, comparable to the original's 4.93% round spread; this small
capture experiment does not establish a general serving speedup. Two prompts
were unchanged, one saved five cycles and one required one extra cycle.
These are the same validation prompts used for checkpoint selection, not an
independent test set. The exported fit remains an experimental draft, not
the default model.

## Reproduce

Use the existing project environment for tokenizer/capture orchestration and
an MLX environment for fitting. Neither Python step loads the target model.
`PRUNED_ASSEMBLY` must be a resolved Splash installation with `model.json`,
`config.json`, `target/`, `draft/` and `tokenizer/`.

```sh
make -j4 build/engine-tests/draft-capture

.venv/bin/python dev/tools/capture_dflash_suite.py \
  --tokenizer "$PRUNED_TARGET" --model "$PRUNED_ASSEMBLY" \
  --output "$NEW_CAPTURE_DIRECTORY" --cycles 48 --max-tokens 256 --hidden yes

.venv/bin/python dev/tools/analyze_dflash_capture.py \
  "$NEW_CAPTURE_DIRECTORY"/*.jsonl --output "$REPORT_PATH"

"$MLX_PYTHON" dev/tools/fit_dflash_selector.py \
  --manifest "$NEW_CAPTURE_DIRECTORY/manifest.json" \
  --draft "$PRUNED_TARGET/draft" --output "$NEW_DRAFT_DIRECTORY" \
  --steps 100 --lr 0.00001 --batch-size 16
```

Prepare a separate local Splash installation with `--draft-model` pointing to
the exported draft, then rerun held-out prompts in the native engine. Match
prompt hashes, generation budgets, KV/state formats and hidden-capture mode
before comparing timings. Use production benchmarks for end-to-end speed.

## Remaining work

1. Select changes using native held-out acceptance and matched-work timing;
   cached CE alone is insufficient. Add independent prompts before further
   hyperparameter selection or making a generalization claim.
2. For natural candidate misses, capture target-tap sequences and adapt the
   context/backbone with token or KL losses. Final draft hidden captures alone
   cannot backpropagate into earlier layers.
3. Reduce the target further with reconstruction/distillation and rerun a
   reproducible target-quality suite against the 64-layer Q4 baseline.
   Draft adaptation improves speculation; it does not restore a pruned
   target's lost coding/reasoning capability.
4. Attribute GPU work by model role and operation before tuning kernels.
   Current profiling groups shared linear pipelines by name and cannot report
   a reliable total draft percentage.
