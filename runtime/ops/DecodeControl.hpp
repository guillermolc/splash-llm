#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/DecodeControl.h"

#include <cstdint>
#include <span>

namespace splash::ops {

struct DecodeControlBuffers final {
  metal::MetalBuffer states;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer retainedCounts;
  metal::MetalBuffer acceptedCounts;
  metal::MetalBuffer draftInputTokens;
  metal::MetalBuffer targetPositions;
  metal::MetalBuffer draftPositions;
  // One cycle's emitted tokens, up to nine per lane. A retained cycle emits
  // its old anchor and retained-1 selected tokens, then its next anchor too
  // when terminal. state.emitted_tokens describes this buffer's live prefix.
  metal::MetalBuffer emittedTokens;
  // Optional per-cycle snapshot for grouped submission. It is written even
  // for an inactive cycle (whose emitted_tokens is zero), so the host can
  // finalize each completed cycle after one grouped command finishes.
  metal::MetalBuffer completedStates{};
};

class DecodeControlPatches final {
public:
  [[nodiscard]] uint32_t count() const noexcept { return count_; }

private:
  metal::MetalBuffer descriptors_;
  uint32_t count_ = 0;
  uint32_t lanes_ = 0;
  uint64_t destinationBytes_ = 0;
  friend class DecodeControl;
};

// Device feedback for repeated text decode graphs. This does not itself
// submit additional model cycles: callers must update every position/state
// parameter and suppress model dispatches once a lane becomes terminal.
class DecodeControl final {
public:
  // Constructs a ready state. The caller has already emitted terminal
  // anchors, so a ready state has at least two output tokens remaining.
  [[nodiscard]] static DecodeControlState initial(
      uint32_t anchor, uint32_t logicalPosition, uint32_t generatedTokens,
      uint32_t maximumNewTokens, uint32_t targetRows,
      const DecodeControlParams &params, int32_t targetRopeOffset = 0);

  // Builds the next cycle's masked draft input and both RoPE position
  // arrays from device state. Inactive lanes write safe zero positions and
  // masked inputs. Every lane keeps its fixed eight-row storage stride.
  static void addPrepare(metal::CommandGraph &graph,
                         const DecodeControlBuffers &buffers,
                         const DecodeControlParams &params, uint32_t lanes);
  // After acceptance: validate the complete retained prefix, advance the
  // anchor, positions and budget, and pack exactly the tokens the host
  // finalizeDecode path emits. The previous state is preserved on failure.
  static void addAdvance(metal::CommandGraph &graph,
                         const DecodeControlBuffers &buffers,
                         const DecodeControlParams &params, uint32_t lanes);

  // Uploads validated immutable patch descriptors once. A graph can reuse
  // them for all subsequent rounds. Destinations must be distinct uint32
  // words inside the packed parameter buffer; invalid metadata is refused
  // before allocating or encoding any work.
  [[nodiscard]] static DecodeControlPatches makePatches(
      metal::MetalBackend &backend, std::span<const DecodeControlPatch> patches,
      uint64_t destinationBytes, uint32_t lanes);
  static void addPatches(metal::CommandGraph &graph, metal::MetalBuffer states,
                         metal::MetalBuffer destination,
                         const DecodeControlPatches &patches);
};

} // namespace splash::ops
