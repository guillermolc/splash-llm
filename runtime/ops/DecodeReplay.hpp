#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops {

struct DecodeReplayCycle final {
  // The first dispatch patches the mutable parameter arena and model
  // grids. Subsequent dispatches are the original body, with every byte
  // parameter promoted to a buffer and every grid gated by device status.
  // Buffer bindings own all storage; no dispatch references caller bytes.
  std::vector<metal::ComputeDispatch> dispatches;
  metal::MetalBuffer parameters;
  uint32_t patchCount = 0;
};

// Prepares an ordinary dense, greedy, unconstrained, fixed-eight-row B1
// graph for execution from GPU-produced feedback. This deliberately rejects
// unsupported pipelines instead of replaying stale host parameters.
class DecodeReplay final {
public:
  // The caller supplies the right GDN state/tape parity for this cycle and
  // enough KV pages for the whole group. It runs DecodeControl::addPrepare
  // before the body and addAdvance after it. The initial graph may preserve
  // its host pending_rows; subsequent cycles patch it from last_retained.
  [[nodiscard]] static DecodeReplayCycle prepare(
      metal::MetalBackend &backend,
      std::span<const metal::ComputeDispatch> body,
      metal::MetalBuffer states, bool patchPendingRows = true);
};

} // namespace splash::ops
