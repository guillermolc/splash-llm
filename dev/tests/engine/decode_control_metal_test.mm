#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/DecodeControl.hpp"
#include "ops/DecodeReplay.hpp"
#include "ops/RowCopy.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/GDN.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/RoPE.h"
#include "metal/abi/Sampling.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::DecodeControl;
using splash::ops::DecodeControlBuffers;
using splash::test::rejects;
using splash::test::require;
constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kPoison = 0xa5a5a5a5;
constexpr DecodeControlParams kParams{1009, 1008, 1006, 1007, kRows};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  auto buffer = splash::test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0xa5, bytes);
  return buffer;
}

template <class T> T *data(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

DecodeControlBuffers buffers(MetalBackend &backend, uint32_t lanes) {
  return {allocate(backend, uint64_t{lanes} * sizeof(DecodeControlState)),
          allocate(backend, uint64_t{lanes} * kRows * 4),
          allocate(backend, uint64_t{lanes} * 4),
          allocate(backend, uint64_t{lanes} * 4),
          allocate(backend, uint64_t{lanes} * kRows * 4),
          allocate(backend, uint64_t{lanes} * kRows * 3 * 4),
          allocate(backend, uint64_t{lanes} * kRows * 4),
          allocate(backend, uint64_t{lanes} * SPLASH_DECODE_EMISSION_ROWS * 4)};
}

// The host runtime's successful-cycle contract, expressed independently of
// the device implementation: output the old anchor and all selected tokens
// except the last, keep that last as the next anchor, and immediately emit
// it too if it is terminal or consumes the final output budget.
std::vector<uint32_t> reference(DecodeControlState &state,
                                std::span<const uint32_t> selected,
                                uint32_t accepted, uint32_t targetRows) {
  std::vector<uint32_t> emitted{state.anchor};
  emitted.insert(emitted.end(), selected.begin(), selected.end() - 1);
  state.anchor = selected.back();
  state.logical_position += selected.size();
  state.generated_tokens += selected.size();
  state.last_retained = selected.size();
  state.last_accepted = accepted;
  ++state.cycles;
  if (state.generated_tokens < state.maximum_new_tokens &&
      (state.anchor == kParams.stop_token_0 || state.anchor == kParams.stop_token_1 ||
       state.maximum_new_tokens - state.generated_tokens == 1)) {
    emitted.push_back(state.anchor);
    ++state.generated_tokens;
    if (state.anchor == kParams.stop_token_0 || state.anchor == kParams.stop_token_1)
      state.status |= SPLASH_DECODE_STOPPED;
  }
  if (state.generated_tokens == state.maximum_new_tokens)
    state.status |= SPLASH_DECODE_BUDGET_EXHAUSTED;
  state.emitted_tokens = emitted.size();
  state.maximum_retained = state.status ? 0 : std::min(
      targetRows, state.maximum_new_tokens - state.generated_tokens);
  return emitted;
}

void checkState(const DecodeControlState &actual,
                const DecodeControlState &expected) {
  require(std::memcmp(&actual, &expected, sizeof(actual)) == 0,
          "GPU decode control differs from the host cycle contract");
}

void checkPrepared(const DecodeControlBuffers &b, uint32_t lane,
                   const DecodeControlState &state) {
  for (uint32_t row = 0; row < kRows; ++row) {
    const uint32_t index = lane * kRows + row;
    const uint32_t position = state.status ? 0 : state.logical_position + row;
    const uint32_t token = !state.status && row == 0 ? state.anchor : kParams.mask_token;
    require(data<uint32_t>(b.draftInputTokens)[index] == token,
            "next draft input did not use the device-produced anchor");
    require(data<uint32_t>(b.draftPositions)[index] == position,
            "next draft RoPE positions did not use the device KV length");
    for (uint32_t axis = 0; axis < 3; ++axis)
      require(data<uint32_t>(b.targetPositions)[index * 3 + axis] ==
                  (state.status ? 0 : uint32_t(int64_t{position} + state.target_rope_offset)),
              "next target RoPE positions differ from the text continuation");
  }
}

void allRetentions(MetalBackend &backend) {
  for (uint32_t rows : {1u, 2u, 4u, 8u}) {
    auto params = kParams;
    params.target_rows = rows;
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      auto b = buffers(backend, lanes);
      for (uint32_t accepted = 0; accepted < rows; ++accepted) {
        for (uint32_t remaining : {2u, 3u, 8u, 9u, 100u}) {
          // No stop, or a stop anywhere inside the otherwise-retained prefix.
          const uint32_t uncut = std::min(accepted + 1, remaining);
          for (uint32_t stopRow = 0; stopRow <= uncut; ++stopRow) {
            std::array<DecodeControlState, kLanes> expected{};
            std::array<std::vector<uint32_t>, kLanes> emitted;
            std::memset(b.emittedTokens.contents(), 0xa5, b.emittedTokens.sizeBytes());
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              auto state = DecodeControl::initial(10 + lane, 701 + lane * 100,
                                                 5, 5 + remaining, rows, params,
                                                 lane % 2 ? -17 : 0);
              data<DecodeControlState>(b.states)[lane] = state;
              auto *selected = data<uint32_t>(b.outputTokens) + lane * kRows;
              for (uint32_t row = 0; row < kRows; ++row)
                selected[row] = 100 + row + lane * 10;
              uint32_t retained = uncut;
              if (stopRow < uncut) {
                selected[stopRow] = lane % 2 ? params.stop_token_1 : params.stop_token_0;
                retained = stopRow + 1;
              }
              data<uint32_t>(b.retainedCounts)[lane] = retained;
              data<uint32_t>(b.acceptedCounts)[lane] = accepted;
              emitted[lane] = reference(state, {selected, retained}, accepted, rows);
              expected[lane] = state;
            }
            CommandGraph graph;
            DecodeControl::addAdvance(graph, b, params, lanes);
            DecodeControl::addPrepare(graph, b, params, lanes);
            static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              checkState(data<DecodeControlState>(b.states)[lane], expected[lane]);
              checkPrepared(b, lane, expected[lane]);
              const auto *actual = data<uint32_t>(b.emittedTokens) +
                                   lane * SPLASH_DECODE_EMISSION_ROWS;
              require(std::equal(emitted[lane].begin(), emitted[lane].end(), actual),
                      "GPU emitted tokens differ from finalizeDecode output");
              for (uint32_t row = emitted[lane].size(); row < SPLASH_DECODE_EMISSION_ROWS; ++row)
                require(actual[row] == kPoison, "GPU emission overwrote its live prefix");
            }
          }
        }
      }
    }
  }
}

void dependentCycles(MetalBackend &backend) {
  auto b = buffers(backend, 1);
  auto expected = DecodeControl::initial(42, 100, 0, 12, kRows, kParams);
  data<DecodeControlState>(b.states)[0] = expected;
  CommandGraph graph;
  std::vector<MetalBuffer> cycleOutputs;
  std::vector<std::vector<uint32_t>> expectedOutputs;
  std::vector<MetalBuffer> snapshots;
  std::vector<DecodeControlState> expectedStates;
  for (uint32_t retained : {3u, 5u, 3u}) {
    b.outputTokens = allocate(backend, kRows * 4);
    b.retainedCounts = allocate(backend, 4);
    b.acceptedCounts = allocate(backend, 4);
    b.emittedTokens = allocate(backend, SPLASH_DECODE_EMISSION_ROWS * 4);
    b.completedStates = allocate(backend, sizeof(DecodeControlState));
    for (uint32_t row = 0; row < kRows; ++row)
      data<uint32_t>(b.outputTokens)[row] = 200 + expected.cycles * 10 + row;
    data<uint32_t>(b.retainedCounts)[0] = retained;
    data<uint32_t>(b.acceptedCounts)[0] = retained - 1;
    DecodeControl::addPrepare(graph, b, kParams, 1);
    DecodeControl::addAdvance(graph, b, kParams, 1);
    expectedOutputs.push_back(reference(expected,
        {data<uint32_t>(b.outputTokens), retained}, retained - 1, kRows));
    cycleOutputs.push_back(b.emittedTokens);
    snapshots.push_back(b.completedStates);
    expectedStates.push_back(expected);
  }
  DecodeControl::addPrepare(graph, b, kParams, 1);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  checkState(data<DecodeControlState>(b.states)[0], expected);
  checkPrepared(b, 0, expected);
  for (size_t cycle = 0; cycle < cycleOutputs.size(); ++cycle) {
    require(std::equal(expectedOutputs[cycle].begin(), expectedOutputs[cycle].end(),
                       data<uint32_t>(cycleOutputs[cycle])),
            "dependent GPU rounds lost an anchor or emitted the wrong tokens");
    checkState(data<DecodeControlState>(snapshots[cycle])[0], expectedStates[cycle]);
  }
  require(expected.status == SPLASH_DECODE_BUDGET_EXHAUSTED && expected.generated_tokens == 12,
          "dependent cycles did not stop at the exact output budget");

  CommandGraph inactive;
  DecodeControl::addAdvance(inactive, b, kParams, 1);
  static_cast<void>(backend.submitCommandAsync(inactive.dispatches()).wait());
  expected.emitted_tokens = 0;
  checkState(data<DecodeControlState>(b.states)[0], expected);
}

void invalidCycles(MetalBackend &backend) {
  auto b = buffers(backend, 1);
  for (uint32_t failure = 0; failure < 7; ++failure) {
    auto state = DecodeControl::initial(42, 100, 0, 12, kRows, kParams);
    std::fill_n(data<uint32_t>(b.outputTokens), kRows, 100);
    data<uint32_t>(b.retainedCounts)[0] = 3;
    data<uint32_t>(b.acceptedCounts)[0] = 2;
    uint32_t status = SPLASH_DECODE_INVALID_COUNT;
    switch (failure) {
    case 0: data<uint32_t>(b.retainedCounts)[0] = 0; break;
    case 1: data<uint32_t>(b.retainedCounts)[0] = kRows + 1; break;
    case 2: data<uint32_t>(b.acceptedCounts)[0] = kRows; break;
    case 3: data<uint32_t>(b.acceptedCounts)[0] = 0; break;
    case 4:
      data<uint32_t>(b.outputTokens)[1] = 0xffffffffu;
      status = SPLASH_DECODE_INVALID_TOKEN; break;
    case 5:
      state.anchor = kParams.vocabulary;
      status = SPLASH_DECODE_INVALID_TOKEN; break;
    case 6:
      state.logical_position = SPLASH_MAXIMUM_CONTEXT_TOKENS;
      status = SPLASH_DECODE_POSITION_OVERFLOW; break;
    }
    data<DecodeControlState>(b.states)[0] = state;
    state.status = status;
    CommandGraph graph;
    DecodeControl::addAdvance(graph, b, kParams, 1);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    checkState(data<DecodeControlState>(b.states)[0], state);
  }
}

void patches(MetalBackend &backend) {
  auto b = buffers(backend, 2);
  data<DecodeControlState>(b.states)[0] = DecodeControl::initial(42, 100, 7, 100, kRows, kParams);
  auto &state = data<DecodeControlState>(b.states)[0];
  state.last_retained = 3;
  state.last_accepted = 2;
  state.cycles = 9;
  data<DecodeControlState>(b.states)[1] = state;
  data<DecodeControlState>(b.states)[1].status = SPLASH_DECODE_STOPPED;
  std::vector<DecodeControlPatch> descriptors;
  for (uint32_t source = 0; source < SPLASH_DECODE_PATCH_SOURCE_COUNT; ++source)
    descriptors.push_back({source, source, 0, 1, 0, 0xffffffffu});
  descriptors.push_back({10, SPLASH_DECODE_PATCH_ACTIVE, 0, 1234, 0, 0xffffffffu});
  descriptors.push_back({11, SPLASH_DECODE_PATCH_ACTIVE, 1, 1234, 0, 0xffffffffu});
  descriptors.push_back({12, SPLASH_DECODE_PATCH_POSITION, 0, 0xffffffffu, 17, 123});
  descriptors.push_back({13, SPLASH_DECODE_PATCH_POSITION, 0, 1, -200, 0xffffffffu});
  descriptors.push_back({14, SPLASH_DECODE_PATCH_POSITION, 0, 1, -17, 0xffffffffu});
  const auto set = DecodeControl::makePatches(backend, descriptors, 16 * 4, 2);
  auto destination = allocate(backend, 16 * 4);
  CommandGraph graph;
  DecodeControl::addPatches(graph, b.states, destination, set);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const std::array<uint32_t, 16> expected{100, 97, 42, 7, 93, 8, 3, 2, 1, 9,
                                          1234, 0, 123, 0, 83, kPoison};
  require(std::equal(expected.begin(), expected.end(), data<uint32_t>(destination)),
          "GPU parameter patch expressions differ from the validated contract");
  for (const auto &bad : std::array{
       DecodeControlPatch{16, 0, 0, 1, 0, 0xffffffffu},
       DecodeControlPatch{0, SPLASH_DECODE_PATCH_SOURCE_COUNT, 0, 1, 0, 0xffffffffu},
       DecodeControlPatch{0, 0, 2, 1, 0, 0xffffffffu}})
    rejects([&] { static_cast<void>(DecodeControl::makePatches(backend, {&bad, 1}, 64, 2)); },
            "invalid decode control patch descriptor", "unsafe parameter patch was accepted");
  const std::array duplicate{descriptors[0], descriptors[0]};
  rejects([&] { static_cast<void>(DecodeControl::makePatches(backend, duplicate, 64, 2)); },
          "invalid decode control patch descriptor", "racing parameter patches were accepted");
}

void extents(MetalBackend &backend) {
  constexpr uint32_t lanes = 3;
  auto b = buffers(backend, lanes);
  b.completedStates = allocate(backend, lanes * sizeof(DecodeControlState));
  for (const auto &[member, bytes, what] :
       std::initializer_list<std::tuple<MetalBuffer DecodeControlBuffers::*, uint64_t, const char *>>{
           {&DecodeControlBuffers::states, lanes * sizeof(DecodeControlState), "decode control state"},
           {&DecodeControlBuffers::draftInputTokens, lanes * kRows * 4, "decode control draft input"},
           {&DecodeControlBuffers::targetPositions, lanes * kRows * 3 * 4, "decode control target position"},
           {&DecodeControlBuffers::draftPositions, lanes * kRows * 4, "decode control draft position"}})
    splash::test::requireExtent(backend, b.*member, bytes, 4, what,
        [&](CommandGraph &graph, const MetalBuffer &view) {
          auto changed = b;
          changed.*member = view;
          DecodeControl::addPrepare(graph, changed, kParams, lanes);
        });
  for (const auto &[member, bytes, what] :
       std::initializer_list<std::tuple<MetalBuffer DecodeControlBuffers::*, uint64_t, const char *>>{
           {&DecodeControlBuffers::states, lanes * sizeof(DecodeControlState), "decode control state"},
           {&DecodeControlBuffers::outputTokens, lanes * kRows * 4, "decode control output token"},
           {&DecodeControlBuffers::retainedCounts, lanes * 4, "decode control retained count"},
           {&DecodeControlBuffers::acceptedCounts, lanes * 4, "decode control accepted count"},
           {&DecodeControlBuffers::emittedTokens, lanes * SPLASH_DECODE_EMISSION_ROWS * 4, "decode control emitted token"}})
    splash::test::requireExtent(backend, b.*member, bytes, 4, what,
        [&](CommandGraph &graph, const MetalBuffer &view) {
          auto changed = b;
          changed.*member = view;
          DecodeControl::addAdvance(graph, changed, kParams, lanes);
        });
  splash::test::requireExtent(backend, b.completedStates,
      lanes * sizeof(DecodeControlState), 4, "decode control completed state",
      [&](CommandGraph &graph, const MetalBuffer &view) {
        auto changed = b;
        changed.completedStates = view;
        DecodeControl::addAdvance(graph, changed, kParams, lanes);
      });
  rejects([&] { static_cast<void>(DecodeControl::initial(42, 100, 0, 1, kRows, kParams)); },
          "invalid initial decode control state", "a terminal initial anchor was accepted");
  CommandGraph graph;
  rejects([&] { DecodeControl::addPrepare(graph, b, kParams, 0); },
          "invalid decode control batch", "an empty GPU control batch was accepted");
  require(graph.empty(), "invalid decode control encoded a dispatch");
}

template <class T> const T &promoted(const splash::metal::ComputeDispatch &dispatch) {
  require(dispatch.bytes.empty(), "replay retained a caller-owned byte parameter");
  require(!dispatch.buffers.empty(), "replay omitted a promoted parameter");
  return *data<T>(dispatch.buffers.back().buffer);
}

void replayParameters(MetalBackend &backend) {
  auto b = buffers(backend, 1);
  auto &state = data<DecodeControlState>(b.states)[0];
  state = DecodeControl::initial(42, 204, 7, 100, kRows, kParams);
  state.last_retained = 5;
  auto dummy = allocate(backend, 64);
  CommandGraph body;
  DraftAttentionBatchParams draft{2048, 1, {}};
  body.add("draft_attention_bf16_split", {dummy}, draft, {8, 1, 4});
  body.add("draft_attention_bf16_reduce", {dummy}, draft, {8, 1, 1});
  body.add("draft_context_kv_commit", {dummy}, DraftContextBatchParams{}, {64, 1, 1});
  GDNDecodeBatchParams gdn{};
  gdn.target_rows = kRows;
  body.add("verify_gdn_fused", {dummy}, gdn, {48, 1, 1});
  std::array<SplashChunkedPrefillParams, kLanes> stores{};
  stores[0].chunk_tokens = kRows;
  body.add("verify_attention_q8_store", {dummy}, stores, {32, 1, 1});
  std::array<SplashVerifyAttentionParams, kLanes> attention{};
  attention[0].rows = kRows;
  body.add("verify_attention_q8_split", {dummy}, attention, {16, 1, 4});
  body.add("verify_attention_reduce", {dummy}, attention, {16, 1, 1});
  AcceptBatchParams accept{};
  accept.target_rows = kRows;
  body.add("decode_accept_dflash", {dummy}, accept, {1, 1, 1});
  SelectorBatchParams selector{};
  selector.lanes = 1;
  selector.device_anchor_mask = 1;
  body.add("draft_select_edges", {dummy}, selector, {7, 1, 1});
  body.add("draft_select_dflash", {dummy}, selector, {1, 1, 1});
  TargetSamplingParams sampling{};
  sampling.rows = kRows;
  sampling.drafted_rows = SPLASH_DRAFT_PROPOSAL_TOKENS;
  body.add("decode_sample_argmax_reduce", {dummy}, sampling, {8, 1, 1});
  body.add("rope_build_tables", {dummy}, RopeTableParams{kRows, kRows}, {2, 1, 1});
  auto replay = splash::ops::DecodeReplay::prepare(backend, body.dispatches(), b.states);
  require(replay.dispatches.size() == body.dispatches().size() + 1,
          "replay omitted body dispatches or device preparation");
  for (uint32_t position : {204u, 512u}) {
    state.logical_position = position;
    static_cast<void>(backend.submit(replay.dispatches[0]));
    require(promoted<DraftAttentionBatchParams>(replay.dispatches[1]).cache_length[0] == position &&
                promoted<DraftAttentionBatchParams>(replay.dispatches[2]).cache_length[0] == position,
            "replay did not patch every draft cache length");
    require(promoted<DraftContextBatchParams>(replay.dispatches[3]).start_position[0] == position,
            "replay did not patch draft context commit position");
    require(promoted<GDNDecodeBatchParams>(replay.dispatches[4]).pending_rows[0] == 5,
            "replay did not patch the retained GDN tape length");
    require(promoted<std::array<SplashChunkedPrefillParams, kLanes>>(replay.dispatches[5])[0].committed_tokens == position,
            "replay did not patch target KV store position");
    for (uint32_t index : {6u, 7u})
      require(promoted<std::array<SplashVerifyAttentionParams, kLanes>>(replay.dispatches[index])[0].committed_tokens == position,
              "replay did not patch every target attention position");
    require(promoted<AcceptBatchParams>(replay.dispatches[8]).remaining[0] == kRows,
            "replay did not patch the target acceptance budget");
    for (size_t index = 1; index < replay.dispatches.size(); ++index) {
      const auto &dispatch = replay.dispatches[index];
      require(dispatch.bytes.empty() && bool(dispatch.indirectThreadgroups.buffer),
              "replay body retained host-owned constants or a direct grid");
      const auto &grid = dispatch.indirectThreadgroups;
      const auto *counts = reinterpret_cast<const uint32_t *>(
          static_cast<const std::byte *>(grid.buffer.contents()) + grid.offsetBytes);
      require(counts[0] == body.dispatches()[index - 1].threadgroups.x &&
                  counts[1] == body.dispatches()[index - 1].threadgroups.y &&
                  counts[2] == body.dispatches()[index - 1].threadgroups.z,
              "active replay grid changed the original geometry");
    }
  }
  state.status = SPLASH_DECODE_STOPPED;
  static_cast<void>(backend.submit(replay.dispatches[0]));
  for (size_t index = 1; index < replay.dispatches.size(); ++index) {
    const auto &grid = replay.dispatches[index].indirectThreadgroups;
    require(*reinterpret_cast<const uint32_t *>(
        static_cast<const std::byte *>(grid.buffer.contents()) + grid.offsetBytes) == 0,
        "terminal replay did not suppress every model dispatch");
  }
  state.status = 0;
  auto initialReplay = splash::ops::DecodeReplay::prepare(
      backend, body.dispatches(), b.states, false);
  static_cast<void>(backend.submit(initialReplay.dispatches[0]));
  require(promoted<GDNDecodeBatchParams>(initialReplay.dispatches[4]).pending_rows[0] == 0,
          "the first replay cycle overwrote its host-provided pending GDN rows");

  CommandGraph unsupported;
  unsupported.add("moe_route_select_f32", {dummy}, {1, 1, 1});
  rejects([&] { static_cast<void>(splash::ops::DecodeReplay::prepare(backend, unsupported.dispatches(), b.states)); },
          "not audited for GPU feedback", "an unaudited replay graph was accepted");
  CommandGraph hostAnchor;
  selector.device_anchor_mask = 0;
  hostAnchor.add("draft_select_edges", {dummy}, selector, {7, 1, 1});
  rejects([&] { static_cast<void>(splash::ops::DecodeReplay::prepare(backend, hostAnchor.dispatches(), b.states)); },
          "device anchors", "a replay graph with a stale host anchor was accepted");
  CommandGraph constrained;
  sampling.constrained_mask = 1;
  constrained.add("decode_sample_argmax_reduce", {dummy}, sampling, {8, 1, 1});
  rejects([&] { static_cast<void>(splash::ops::DecodeReplay::prepare(backend, constrained.dispatches(), b.states)); },
          "unconstrained greedy", "a constrained replay graph was accepted");
  CommandGraph narrow;
  accept.target_rows = 4;
  narrow.add("decode_accept_dflash", {dummy}, accept, {1, 1, 1});
  rejects([&] { static_cast<void>(splash::ops::DecodeReplay::prepare(backend, narrow.dispatches(), b.states)); },
          "eight-row acceptance", "a narrow replay graph was accepted without an audit");
}

void replayDispatchSuppression(MetalBackend &backend) {
  auto b = buffers(backend, 1);
  auto &state = data<DecodeControlState>(b.states)[0];
  state = DecodeControl::initial(42, 204, 7, 100, kRows, kParams);
  auto source = allocate(backend, 4 * sizeof(uint16_t));
  auto destination = allocate(backend, source.sizeBytes());
  std::iota(data<uint16_t>(source), data<uint16_t>(source) + 4, uint16_t{10});
  CommandGraph body;
  splash::ops::RowCopy::add(body, source, {0, 4, 0}, destination, {0, 4, 0}, 1, 4);
  auto replay = splash::ops::DecodeReplay::prepare(backend, body.dispatches(), b.states);
  static_cast<void>(backend.submitCommandAsync(replay.dispatches).wait());
  require(std::memcmp(source.contents(), destination.contents(), source.sizeBytes()) == 0,
          "promoted constant buffers changed an active replay dispatch");
  state.status = SPLASH_DECODE_STOPPED;
  std::memset(destination.contents(), 0xa5, destination.sizeBytes());
  static_cast<void>(backend.submitCommandAsync(replay.dispatches).wait());
  for (uint32_t index = 0; index < 4; ++index)
    require(data<uint16_t>(destination)[index] == 0xa5a5,
            "a terminal replay still executed a model dispatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: decode-control METALLIB");
    MetalBackend backend(argv[1]);
    allRetentions(backend);
    dependentCycles(backend);
    invalidCycles(backend);
    patches(backend);
    extents(backend);
    replayParameters(backend);
    replayDispatchSuppression(backend);
    std::cout << "decode_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "decode_control_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
