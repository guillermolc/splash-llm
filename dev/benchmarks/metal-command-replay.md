# Metal command replay

`MetalBackend::setCommandReplayEnabled(true)` enables a bounded cache of
preencoded Metal compute indirect command buffers (ICBs). It is disabled by
default. The ordinary command path remains available for comparison and for
commands containing shared-event steps or dispatch profiling.

## Implementation

- Cache keys include pipelines, grid dimensions, threadgroup dimensions,
  buffer allocation/view identities, indirect-grid locations and byte-binding
  sizes. Byte-binding values are copied into an aligned shared constant buffer
  before each submission, so changing positions does not require a new ICB.
- The first encounter of a graph uses ordinary encoding; a second encounter
  builds its ICB. The LRU cache holds at most eight graphs and 64 MiB of ICB,
  constant and grid-update buffer storage. This limit excludes pipeline states,
  host metadata and the existing model/state buffers the ICB binds.
- Every ICB dispatch has a barrier preserving the ordinary backend's sequential
  dispatch semantics, including buffers reached through GPU addresses.
- For GPU-written indirect grids, the graph is split into consecutive static
  and indirect ranges. One GPU kernel updates the next indirect range's ICB
  dispatch dimensions from the original grid buffers. Separate compute encoders
  order the grid producer, ICB update and ICB execution without CPU readback.
  Zero grids skip dispatches.
- A ticket retains its ICB and copied arguments until completion. An overlapping
  submission is refused before it can overwrite in-flight constants. Cache
  eviction is safe while a command is in flight.
- `releaseMemory()` clears the cache before detaching native buffer storage.
  Reclamation paths that drop buffer owners without calling `releaseMemory()`
  must call `clearCommandReplayCache()` first. A cached graph otherwise retains
  its bound allocations until eviction or cache clearing.
- `commandReplayStats()` exposes hits, misses, builds, bypasses, evictions,
  current cache storage and the last bypass reason. These reads follow the
  backend's single submitting thread rule.

## Correctness checks

The Metal backend suite covers changing byte arguments; dependencies across
different pipelines; buffer views and grid shape changes; GPU-written grids
changed twice in one command; dependent indirect dispatches; zero grids; event
and profiling fallback; release/restore; bounded LRU eviction; and in-flight
resource lifetime.

```sh
make build/engine-tests/metal-backend build/engine-tests/metal-backend.metallib
build/engine-tests/metal-backend build/engine-tests/metal-backend.metallib
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  build/engine-tests/metal-backend build/engine-tests/metal-backend.metallib \
  --replay-validation
```

`--replay-validation` runs the ICB dependency, indirect-grid and cache tests.
The ordinary full suite additionally uses native Objective-C method swizzling
to hold commands and test watchdog/lifetime behavior. Those interception tests
are incompatible with the Metal validation layer's command/event wrappers.

On an Apple M5 Max with 40 GPU cores, both the ordinary full suite and the
isolated ICB suite with API and shader validation passed on October 24, 2026.

## Model-free microbenchmark

```sh
make metal-command-replay-benchmark
```

This runs 256 dependent additions over 64 words, with three warmup commands
per block and 24 measured commands per block in ABBA order. It checks every
output word and reports the median of 48 samples per mode as JSON. Host time
includes command preparation and submission; wall time also includes waiting
for completion and normal memory telemetry. GPU time comes from the Metal
command buffer. The benchmark loads no model.

The preserved benchmark target on this M5 Max measured the following values:

| Median time | Ordinary encoding | Cached ICB |
| --- | ---: | ---: |
| Host submission | 0.060937 ms | 0.032875 ms |
| Complete submission and wait | 0.655812 ms | 0.878688 ms |
| GPU | 0.396875 ms | 0.704125 ms |

Host submission was about 46% lower, but complete wall time was about 34%
higher in this run. A preceding standalone run measured 1.135333 ms ordinary
versus 0.912667 ms replay wall time, showing that these tiny GPU timings vary
between runs. Explicit ICB barriers and GPU scheduling can outweigh the host
savings. Replay therefore remains opt-in, without an accepted model-level
speed claim.

These figures describe tiny dispatches, not Qwen throughput. A real model
comparison must include the full runtime, dynamic-grid update overhead,
acceptance behavior, prefill and output equivalence. Do not transfer the
microbenchmark's relative improvement to model token rates.
