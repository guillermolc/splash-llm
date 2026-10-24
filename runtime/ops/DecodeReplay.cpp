#include "ops/DecodeReplay.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/GDN.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/RoPE.h"
#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"
#include "ops/DecodeControl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace splash::ops {
namespace {

using metal::BytesBinding;
using metal::ComputeDispatch;

[[noreturn]] void unsupported(const ComputeDispatch &dispatch,
                             const char *reason) {
  throw std::invalid_argument("decode replay unsupported pipeline " +
                              dispatch.pipelineName + ": " + reason);
}

template <class T> T parameter(const ComputeDispatch &dispatch) {
  if (dispatch.bytes.size() != 1 || !dispatch.bytes[0].data ||
      dispatch.bytes[0].sizeBytes != sizeof(T))
    unsupported(dispatch, "unexpected parameter ABI");
  T value;
  std::memcpy(&value, dispatch.bytes[0].data, sizeof(value));
  return value;
}

// All offsets are relative to the dispatch's sole parameter struct. Arrays
// retain their maximum-four-lane ABI; only lane zero is active in this path.
std::vector<uint32_t> dynamicOffsets(const ComputeDispatch &dispatch,
                                    bool patchPendingRows,
                                    std::vector<uint32_t> &sources) {
  const std::string_view name = dispatch.pipelineName;
  sources.clear();
  if (name == "draft_attention_bf16_split" || name == "draft_attention_bf16_reduce") {
    const auto params = parameter<DraftAttentionBatchParams>(dispatch);
    if (params.lanes != 1) unsupported(dispatch, "only B1 is supported");
    sources.push_back(SPLASH_DECODE_PATCH_POSITION);
    return {offsetof(DraftAttentionBatchParams, cache_length)};
  }
  if (name == "draft_context_kv_commit") {
    static_cast<void>(parameter<DraftContextBatchParams>(dispatch));
    sources.push_back(SPLASH_DECODE_PATCH_POSITION);
    return {offsetof(DraftContextBatchParams, start_position)};
  }
  if (name.starts_with("verify_gdn_fused")) {
    const auto params = parameter<GDNDecodeBatchParams>(dispatch);
    if (params.target_rows != SPLASH_TARGET_VERIFY_ROWS)
      unsupported(dispatch, "only eight target rows are supported");
    if (patchPendingRows) {
      sources.push_back(SPLASH_DECODE_PATCH_RETAINED);
      return {offsetof(GDNDecodeBatchParams, pending_rows)};
    }
    return {};
  }
  if (name == "verify_attention_q8_store" || name == "verify_attention_q8_store_kv2_g8" ||
      name == "verify_attention_bf16_store" || name == "verify_attention_bf16_store_kv2_g8") {
    const auto params = parameter<std::array<SplashChunkedPrefillParams, SPLASH_MAXIMUM_BATCH_WIDTH>>(dispatch);
    if (params[0].chunk_tokens != SPLASH_TARGET_VERIFY_ROWS)
      unsupported(dispatch, "only eight target rows are supported");
    sources.push_back(SPLASH_DECODE_PATCH_POSITION);
    return {offsetof(SplashChunkedPrefillParams, committed_tokens)};
  }
  if (name == "verify_attention_q8_split" || name == "verify_attention_q8_split_kv2_g8" ||
      name == "verify_attention_bf16_split" || name == "verify_attention_bf16_split_kv2_g8" ||
      name == "verify_attention_reduce" || name == "verify_attention_reduce_kv2_g8") {
    const auto params = parameter<std::array<SplashVerifyAttentionParams, SPLASH_MAXIMUM_BATCH_WIDTH>>(dispatch);
    if (params[0].rows != SPLASH_TARGET_VERIFY_ROWS)
      unsupported(dispatch, "only eight target rows are supported");
    sources.push_back(SPLASH_DECODE_PATCH_POSITION);
    return {offsetof(SplashVerifyAttentionParams, committed_tokens)};
  }
  if (name == "decode_accept_dflash") {
    const auto params = parameter<AcceptBatchParams>(dispatch);
    if (params.sampling_mask || params.target_rows != SPLASH_TARGET_VERIFY_ROWS)
      unsupported(dispatch, "only greedy eight-row acceptance is supported");
    sources.push_back(SPLASH_DECODE_PATCH_MAXIMUM_RETAINED);
    return {offsetof(AcceptBatchParams, remaining)};
  }
  if (name == "draft_select_edges" || name == "draft_select_dflash") {
    const auto params = parameter<SelectorBatchParams>(dispatch);
    if (params.lanes != 1 || params.sampling_mask || params.device_anchor_mask != 1)
      unsupported(dispatch, "selection requires greedy B1 device anchors");
    return {};
  }
  if (name == "decode_sample_argmax_sharded" || name == "decode_sample_argmax_reduce") {
    const auto params = parameter<TargetSamplingParams>(dispatch);
    if (params.sampling_mask || params.constrained_mask ||
        params.rows != SPLASH_TARGET_VERIFY_ROWS || params.logits_row != 0 ||
        params.drafted_rows != SPLASH_DRAFT_PROPOSAL_TOKENS)
      unsupported(dispatch, "only unconstrained greedy eight-row verify is supported");
    return {};
  }
  if (name == "rope_build_tables") {
    const auto params = parameter<RopeTableParams>(dispatch);
    if (params.target_rows != SPLASH_TARGET_VERIFY_ROWS || params.draft_rows != SPLASH_DRAFT_QUERY_ROWS)
      unsupported(dispatch, "only B1 eight-row RoPE tables are supported");
    return {};
  }
  if (name == "gguf_float_bf16" || name == "gguf_float_f32") {
    const auto params = parameter<GgufFloatParams>(dispatch);
    if (params.rows != SPLASH_TARGET_VERIFY_ROWS)
      unsupported(dispatch, "only B1 eight-row float projections are supported");
    return {};
  }
  // The following families consume only tensor shape, weight-layout,
  // normalization or convolution parameters. Their decode implementations
  // do not embed sequence position, retained count or pending anchor.
  if (name.starts_with("gguf_embed_") || name.starts_with("gguf_decode_") ||
      name.starts_with("norm_rms") || name.starts_with("verify_attention_qkv") ||
      name.starts_with("verify_attention_gate") || name == "gguf_rotate" ||
      name == "decode_linear_gguf_prepare" || name == "draft_conv" ||
      name == "draft_conv_h2048" || name == "draft_attention_qkv" ||
      name == "draft_attention_reorder" || name == "draft_select_top16_sharded" ||
      name == "verify_input_tokens" || name == "copy_rows_bf16")
    return {};
  unsupported(dispatch, "not audited for GPU feedback");
}

uint64_t aligned(uint64_t bytes) {
  if (bytes > std::numeric_limits<uint64_t>::max() - 255)
    throw std::overflow_error("decode replay parameter arena overflow");
  return (bytes + 255) / 256 * 256;
}

} // namespace

DecodeReplayCycle DecodeReplay::prepare(
    metal::MetalBackend &backend, std::span<const ComputeDispatch> body,
    metal::MetalBuffer states, bool patchPendingRows) {
  if (body.empty())
    throw std::invalid_argument("decode replay requires a nonempty graph");
  requireBytes(states, sizeof(DecodeControlState), "decode replay state");
  struct Payload final { BytesBinding bytes; uint64_t offset; };
  std::vector<std::vector<Payload>> payloads(body.size());
  std::vector<uint64_t> gridOffsets;
  std::vector<DecodeControlPatch> descriptors;
  uint64_t arenaBytes = 0;
  std::vector<uint32_t> sources;
  for (size_t index = 0; index < body.size(); ++index) {
    const auto &dispatch = body[index];
    if (dispatch.indirectThreadgroups.buffer)
      unsupported(dispatch, "an existing indirect grid cannot be replaced");
    const auto offsets = dynamicOffsets(dispatch, patchPendingRows, sources);
    for (const auto &bytes : dispatch.bytes) {
      if (!bytes.data || !bytes.sizeBytes || bytes.sizeBytes > 4096)
        unsupported(dispatch, "invalid byte parameter binding");
      if (std::any_of(dispatch.buffers.begin(), dispatch.buffers.end(),
                      [&](const auto &binding) { return binding.index == bytes.index; }))
        unsupported(dispatch, "overlapping buffer and byte bindings");
      payloads[index].push_back({bytes, arenaBytes});
      for (size_t field = 0; field < offsets.size(); ++field) {
        if (offsets[field] + sizeof(uint32_t) > bytes.sizeBytes)
          unsupported(dispatch, "dynamic field is outside the parameter ABI");
        const uint64_t word = (arenaBytes + offsets[field]) / sizeof(uint32_t);
        if (word > std::numeric_limits<uint32_t>::max())
          throw std::overflow_error("decode replay parameter arena exceeds patch ABI");
        descriptors.push_back({static_cast<uint32_t>(word), sources[field], 0, 1, 0,
                               std::numeric_limits<uint32_t>::max()});
      }
      arenaBytes += aligned(bytes.sizeBytes);
    }
    if (!dispatch.threadgroups.x || !dispatch.threadgroups.y || !dispatch.threadgroups.z ||
        dispatch.threadgroups.x > std::numeric_limits<uint32_t>::max() ||
        dispatch.threadgroups.y > std::numeric_limits<uint32_t>::max() ||
        dispatch.threadgroups.z > std::numeric_limits<uint32_t>::max())
      unsupported(dispatch, "grid is outside the indirect dispatch ABI");
    gridOffsets.push_back(arenaBytes);
    if (arenaBytes / 4 > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("decode replay grids exceed patch ABI");
    descriptors.push_back({static_cast<uint32_t>(arenaBytes / 4),
                           SPLASH_DECODE_PATCH_ACTIVE, 0,
                           static_cast<uint32_t>(dispatch.threadgroups.x), 0,
                           std::numeric_limits<uint32_t>::max()});
    arenaBytes += 256;
  }
  DecodeReplayCycle result;
  result.parameters = backend.allocateBuffer(arenaBytes, metal::BufferStorage::Shared,
                                              "decode replay parameters");
  std::memset(result.parameters.contents(), 0, arenaBytes);
  result.patchCount = static_cast<uint32_t>(descriptors.size());
  result.dispatches.reserve(body.size() + 1);
  const auto patches = DecodeControl::makePatches(backend, descriptors, arenaBytes, 1);
  metal::CommandGraph preparation;
  DecodeControl::addPatches(preparation, states, result.parameters, patches);
  // The patch dispatch has one tiny static count parameter; promoting it
  // too leaves no pointer into preparation's short-lived byte storage.
  auto patchDispatch = preparation.dispatches()[0];
  for (const auto &bytes : patchDispatch.bytes) {
    auto buffer = backend.allocateBuffer(bytes.sizeBytes, metal::BufferStorage::Shared,
                                           "decode replay patch count");
    std::memcpy(buffer.contents(), bytes.data, bytes.sizeBytes);
    patchDispatch.buffers.push_back({bytes.index, std::move(buffer)});
  }
  patchDispatch.bytes.clear();
  result.dispatches.push_back(std::move(patchDispatch));
  auto *arena = static_cast<std::byte *>(result.parameters.contents());
  for (size_t index = 0; index < body.size(); ++index) {
    auto dispatch = body[index];
    for (const auto &payload : payloads[index]) {
      std::memcpy(arena + payload.offset, payload.bytes.data, payload.bytes.sizeBytes);
      dispatch.buffers.push_back({payload.bytes.index,
          backend.view(result.parameters, payload.offset, payload.bytes.sizeBytes)});
    }
    dispatch.bytes.clear();
    const std::array<uint32_t, 3> grid{
        static_cast<uint32_t>(dispatch.threadgroups.x),
        static_cast<uint32_t>(dispatch.threadgroups.y),
        static_cast<uint32_t>(dispatch.threadgroups.z)};
    std::memcpy(arena + gridOffsets[index], grid.data(), sizeof(grid));
    dispatch.indirectThreadgroups = {result.parameters, gridOffsets[index]};
    result.dispatches.push_back(std::move(dispatch));
  }
  return result;
}

} // namespace splash::ops
