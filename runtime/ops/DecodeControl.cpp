#include "ops/DecodeControl.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace splash::ops {
namespace {

void validate(const DecodeControlParams &params, uint32_t lanes) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !params.vocabulary ||
      params.mask_token >= params.vocabulary || !params.target_rows ||
      params.target_rows > SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid decode control batch");
}

} // namespace

DecodeControlState DecodeControl::initial(
    uint32_t anchor, uint32_t logicalPosition, uint32_t generatedTokens,
    uint32_t maximumNewTokens, uint32_t targetRows,
    const DecodeControlParams &params, int32_t targetRopeOffset) {
  validate(params, 1);
  const int64_t firstRope = int64_t{logicalPosition} + targetRopeOffset;
  if (anchor >= params.vocabulary || anchor == params.stop_token_0 ||
      anchor == params.stop_token_1 || generatedTokens >= maximumNewTokens ||
      maximumNewTokens - generatedTokens < 2 || !targetRows ||
      targetRows > SPLASH_TARGET_VERIFY_ROWS || targetRows != params.target_rows ||
      logicalPosition > SPLASH_MAXIMUM_CONTEXT_TOKENS || firstRope < 0 ||
      firstRope + SPLASH_TARGET_VERIFY_ROWS - 1 >
          std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid initial decode control state");
  return {anchor, logicalPosition, generatedTokens, maximumNewTokens,
          targetRopeOffset, 0, 0, 0, 0, 0,
          std::min(targetRows, maximumNewTokens - generatedTokens), 0};
}

void DecodeControl::addPrepare(metal::CommandGraph &graph,
                               const DecodeControlBuffers &buffers,
                               const DecodeControlParams &params,
                               uint32_t lanes) {
  validate(params, lanes);
  const uint64_t rows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.states, uint64_t{lanes} * sizeof(DecodeControlState),
               "decode control state");
  requireBytes(buffers.draftInputTokens, rows * sizeof(uint32_t),
               "decode control draft input");
  requireBytes(buffers.targetPositions, rows * 3 * sizeof(uint32_t),
               "decode control target position");
  requireBytes(buffers.draftPositions, rows * sizeof(uint32_t),
               "decode control draft position");
  graph.add("decode_control_prepare",
            {buffers.states, buffers.draftInputTokens, buffers.targetPositions,
             buffers.draftPositions},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void DecodeControl::addAdvance(metal::CommandGraph &graph,
                               const DecodeControlBuffers &buffers,
                               const DecodeControlParams &params,
                               uint32_t lanes) {
  validate(params, lanes);
  requireBytes(buffers.states, uint64_t{lanes} * sizeof(DecodeControlState),
               "decode control state");
  requireBytes(buffers.outputTokens,
               uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS * sizeof(uint32_t),
               "decode control output token");
  requireBytes(buffers.retainedCounts, uint64_t{lanes} * sizeof(uint32_t),
               "decode control retained count");
  requireBytes(buffers.acceptedCounts, uint64_t{lanes} * sizeof(uint32_t),
               "decode control accepted count");
  requireBytes(buffers.emittedTokens,
               uint64_t{lanes} * SPLASH_DECODE_EMISSION_ROWS * sizeof(uint32_t),
               "decode control emitted token");
  if (buffers.completedStates)
    requireBytes(buffers.completedStates, uint64_t{lanes} * sizeof(DecodeControlState),
                 "decode control completed state");
  graph.add("decode_control_advance",
            {buffers.states, buffers.outputTokens, buffers.retainedCounts,
             buffers.acceptedCounts, buffers.emittedTokens,
             buffers.completedStates ? buffers.completedStates : buffers.states},
            params, {lanes, 1, 1}, {1, 1, 1});
}

DecodeControlPatches DecodeControl::makePatches(
    metal::MetalBackend &backend, std::span<const DecodeControlPatch> patches,
    uint64_t destinationBytes, uint32_t lanes) {
  if (patches.empty() || patches.size() > std::numeric_limits<uint32_t>::max() ||
      !destinationBytes || destinationBytes % sizeof(uint32_t) || !lanes ||
      lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid decode control patches");
  std::unordered_set<uint32_t> destinations;
  for (const auto &patch : patches) {
    if (uint64_t{patch.destination_word} >= destinationBytes / sizeof(uint32_t) ||
        patch.source >= SPLASH_DECODE_PATCH_SOURCE_COUNT || patch.lane >= lanes ||
        !destinations.insert(patch.destination_word).second)
      throw std::invalid_argument("invalid decode control patch descriptor");
  }
  DecodeControlPatches result;
  result.count_ = static_cast<uint32_t>(patches.size());
  result.lanes_ = lanes;
  result.destinationBytes_ = destinationBytes;
  result.descriptors_ = backend.allocateBuffer(
      patches.size_bytes(), metal::BufferStorage::Shared, "decode control patches");
  std::memcpy(result.descriptors_.contents(), patches.data(), patches.size_bytes());
  return result;
}

void DecodeControl::addPatches(metal::CommandGraph &graph,
                               metal::MetalBuffer states,
                               metal::MetalBuffer destination,
                               const DecodeControlPatches &patches) {
  if (!patches.count_)
    throw std::invalid_argument("invalid decode control patches");
  requireBytes(states, uint64_t{patches.lanes_} * sizeof(DecodeControlState),
               "decode control state");
  requireBytes(destination, patches.destinationBytes_, "decode control patch destination");
  graph.add("decode_control_patch", {std::move(states), patches.descriptors_,
                                     std::move(destination)},
            DecodeControlPatchParams{patches.count_},
            {(uint64_t{patches.count_} + 63) / 64, 1, 1}, {64, 1, 1});
}

} // namespace splash::ops
