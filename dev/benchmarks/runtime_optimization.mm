// Sequential native A/B harness. Loads one immutable target/draft pair and
// creates fresh request state for each arm; no training or model conversion.
//
// runtime-optimization METALLIB MODEL_ROOT CASES_FILE OUTPUT_JSONL
//   [--modes baseline,lookup,rows1,rows2,rows4,auto,control,replay,combined,all]
//   [--trials N] [--warmup-tokens N]
//
// CASES_FILE is little-endian: uint32 case_count; then for each case:
// uint32 name_bytes, UTF-8 name, uint32 prompt_count, uint32[prompt_count],
// uint32 max_new_tokens, float32 temperature, uint64 seed.

#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "ops/PageStorage.hpp"

#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t kAllocationLimit = 32ULL << 30;
// The same harness can link against the unmodified baseline checkout.
// Only observability of newly introduced features is disabled there.
#ifdef SPLASH_BENCH_BASELINE_COMPAT
struct ReplayTelemetry {
  uint64_t hits = 0, builds = 0, bypasses = 0, cachedBytes = 0;
};
struct OptimizationTelemetry {
  uint64_t lookupCycles = 0, lookupProposals = 0, lookupAccepted = 0;
  uint64_t neuralDraftCycles = 0, targetOnlyCycles = 0;
  uint64_t gpuControlledCycles = 0, groupedGpuCommands = 0;
  std::array<uint64_t, 4> targetRowCycles{};
};
void clearReplay(metal::MetalBackend &) {}
ReplayTelemetry replayTelemetry(const metal::MetalBackend &) { return {}; }
OptimizationTelemetry optimizationTelemetry(const model::Runtime &) { return {}; }
#else
void clearReplay(metal::MetalBackend &backend) { backend.clearCommandReplayCache(); }
auto replayTelemetry(const metal::MetalBackend &backend) { return backend.commandReplayStats(); }
auto optimizationTelemetry(const model::Runtime &executor) { return executor.optimizationTelemetry(); }
#endif
struct Case {
  std::string name;
  std::vector<uint32_t> prompt;
  uint32_t maximumNewTokens = 0;
  float temperature = 0;
  uint64_t seed = 0;
};

template <class T> T readScalar(std::istream &input) {
  static_assert(std::endian::native == std::endian::little);
  T value{};
  if (!input.read(reinterpret_cast<char *>(&value), sizeof(value)))
    throw std::invalid_argument("truncated binary case file");
  return value;
}
std::vector<Case> readCases(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::invalid_argument("cannot open binary case file");
  const uint32_t count = readScalar<uint32_t>(input);
  if (!count || count > 1024) throw std::invalid_argument("case_count must be in 1..1024");
  std::vector<Case> cases;
  for (uint32_t index = 0; index < count; ++index) {
    Case item;
    const uint32_t nameBytes = readScalar<uint32_t>(input);
    if (!nameBytes || nameBytes > 4096) throw std::invalid_argument("invalid case name size");
    item.name.resize(nameBytes);
    if (!input.read(item.name.data(), nameBytes)) throw std::invalid_argument("truncated case name");
    const uint32_t tokens = readScalar<uint32_t>(input);
    if (!tokens || tokens > 131072) throw std::invalid_argument("prompt_count must be in 1..131072");
    item.prompt.resize(tokens);
    if (!input.read(reinterpret_cast<char *>(item.prompt.data()), uint64_t{tokens} * sizeof(uint32_t)))
      throw std::invalid_argument("truncated prompt tokens");
    item.maximumNewTokens = readScalar<uint32_t>(input);
    item.temperature = readScalar<float>(input);
    item.seed = readScalar<uint64_t>(input);
    if (!item.maximumNewTokens || item.maximumNewTokens > 4096 ||
        !std::isfinite(item.temperature) || item.temperature < 0 || item.temperature > 2)
      throw std::invalid_argument("invalid generation parameters");
    SamplingParameters policy;
    policy.temperature = item.temperature;
    if (const auto error = policy.validationError()) throw std::invalid_argument(std::string(*error));
    cases.push_back(std::move(item));
  }
  if (input.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("trailing bytes in case file");
  return cases;
}
uint32_t count(std::string_view value) {
  uint32_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !parsed)
    throw std::invalid_argument("option requires a positive integer");
  return parsed;
}
std::vector<std::string> splitModes(std::string_view text) {
  std::vector<std::string> modes;
  constexpr std::array<std::string_view, 10> accepted{
      "baseline", "lookup", "rows1", "rows2", "rows4", "auto", "control", "replay", "combined", "all"};
  while (!text.empty()) {
    const auto part = text.substr(0, text.find(','));
    if (std::find(accepted.begin(), accepted.end(), part) == accepted.end())
      throw std::invalid_argument("unknown mode: " + std::string(part));
    if (std::find(modes.begin(), modes.end(), part) != modes.end())
      throw std::invalid_argument("duplicate benchmark mode");
    modes.emplace_back(part);
    if (part.size() == text.size()) break;
    text.remove_prefix(part.size() + 1);
  }
  if (modes.empty()) throw std::invalid_argument("empty benchmark modes");
  return modes;
}
void configure(std::string_view mode) {
  const auto set = [](const char *name, const char *value) {
    if (setenv(name, value, 1)) throw std::runtime_error("setenv failed");
  };
  set("SPLASH_RUNTIME_OPTIMIZATIONS", "0");
  set("SPLASH_PROMPT_LOOKUP", mode == "lookup" || mode == "combined" || mode == "all" ? "1" : "0");
  set("SPLASH_GPU_DECODE_CONTROL", mode == "control" || mode == "combined" || mode == "all" ? "1" : "0");
  set("SPLASH_COMMAND_REPLAY", mode == "replay" || mode == "combined" || mode == "all" ? "1" : "0");
  const char *rows = "8";
  if (mode == "rows1") rows = "1";
  if (mode == "rows2") rows = "2";
  if (mode == "rows4") rows = "4";
  if (mode == "auto" || mode == "all") rows = "auto";
  set("SPLASH_VERIFY_ROWS", rows);
}
uint64_t footprintBytes() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t size = TASK_VM_INFO_COUNT;
  return task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &size) == KERN_SUCCESS
      ? info.phys_footprint : 0;
}
uint64_t swapBytes() {
  struct xsw_usage value{};
  size_t bytes = sizeof(value);
  return sysctlbyname("vm.swapusage", &value, &bytes, nullptr, 0) == 0 ? value.xsu_used : 0;
}
void memoryGate(const metal::MetalBackend &backend, uint64_t extraBytes = 0) {
  const auto memory = backend.memoryStats();
  const uint64_t charged = std::max(memory.allocatedBytes, memory.deviceCurrentAllocatedBytes);
  if (extraBytes > kAllocationLimit || charged > kAllocationLimit - extraBytes)
    throw std::runtime_error("benchmark stopped at the 32 GiB Metal allocation limit");
  if (footprintBytes() > (48ULL << 30))
    throw std::runtime_error("benchmark stopped at the 48 GiB process footprint limit");
  const auto host = engine::queryHostMemory();
  const uint64_t reserve = backend.capabilities().physicalMemoryBytes * 15 / 100;
  if (!host || host->available <= reserve || extraBytes > host->available - reserve)
    throw std::runtime_error("benchmark stopped to preserve 15 percent available host memory");
  if (engine::querySystemMemoryPressure().value_or(engine::MemoryPressure::Critical) != engine::MemoryPressure::Normal)
    throw std::runtime_error("benchmark stopped because host memory pressure is not normal");
}
template <class T> void array(std::ostream &out, std::span<const T> values) {
  out << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) out << ',';
    out << values[index];
  }
  out << ']';
}

std::string run(metal::MetalBackend &backend, const model::LoadedModel &loaded,
                const ops::ExecutionPlans &operators, kv::PageStorage &pages,
                engine::MemoryGovernor &governor, const Case &item,
                std::string_view mode, uint32_t trial, uint32_t order,
                bool warmup, uint32_t warmupTokens) {
  configure(mode);
  clearReplay(backend);
  memoryGate(backend);
  const auto setupStarted = Clock::now();
  const uint64_t swapBefore = swapBytes();
  model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                 loaded.stateLayout(ops::GdnStateFormat::Float32), nullptr);
  model::Runtime executor({backend, loaded, pages, states, operators});
  const auto replayBefore = replayTelemetry(backend);
  engine::EngineRequest request;
  request.id = 1;
  request.prompt = item.prompt;
  request.maxNewTokens = warmup ? std::min(item.maximumNewTokens, warmupTokens) : item.maximumNewTokens;
  request.sampling.temperature = item.temperature;
  request.sampling.seed = item.seed;
  executor.beginColdRequest(request.modelView(), 0);
  const uint32_t neededPages = (uint64_t{item.prompt.size()} + request.maxNewTokens + 16 + kv::kPageTokens - 1) /
                               kv::kPageTokens;
  std::vector<uint32_t> table(neededPages);
  std::iota(table.begin(), table.end(), 0U);
  std::vector<uint32_t> output, retainedCounts, acceptedCounts, draftedCounts;
  std::vector<double> cycleGpuSeconds, cycleWallSeconds;
  bool finished = false;
  uint64_t logicalPosition = 0;
  const auto started = Clock::now();
  const double setupSeconds = std::chrono::duration<double>(started - setupStarted).count();
  for (uint32_t offset = 0; offset < item.prompt.size();) {
    memoryGate(backend);
    const uint32_t tokens = std::min<uint32_t>(model::ExecutionLimits::prefillTokenBudget,
                                               static_cast<uint32_t>(item.prompt.size()) - offset);
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{1, tokens, offset}},
                           .decodeStage = DecodeStage::Regular};
    ModelBatchItem batch{1, offset, tokens, table, 1};
    batch.inputTokens = std::span(item.prompt).subspan(offset, tokens);
    const auto results = executor.prefill(plan, {&batch, 1});
    if (results.size() != 1 || results[0].consumedPromptTokens != tokens)
      throw std::runtime_error("prefill returned an incorrect row count");
    const auto &result = results[0];
    if (!result.failure.empty()) throw std::runtime_error(result.failure);
    output.insert(output.end(), result.outputTokens.begin(), result.outputTokens.end());
    finished = result.finished;
    offset += tokens;
  }
  logicalPosition = item.prompt.size();
  const auto prefilled = Clock::now();
  const uint64_t commandsBefore = metal::BackendInstrumentation::submittedCommands(backend);
  uint64_t peakAllocated = backend.memoryStats().allocatedBytes;
  uint64_t peakFootprint = footprintBytes();
  while (!finished && output.size() < request.maxNewTokens) {
    memoryGate(backend);
    const auto cycleStarted = Clock::now();
    BatchPlan plan{.kind = WorkKind::Decode, .items = {{1, 0, 0}},
                           .decodeStage = DecodeStage::Regular};
    ModelBatchItem batch{1, logicalPosition, 0, table, 1};
    const auto results = executor.decode(plan, {&batch, 1});
    if (results.size() != 1) throw std::runtime_error("decode returned an incorrect lane count");
    const auto &result = results[0];
    if (!result.failure.empty()) throw std::runtime_error(result.failure);
    if (result.outputTokens.empty()) throw std::runtime_error("decode made no progress");
    const uint32_t retained = static_cast<uint32_t>(result.outputTokens.size()) - result.outputTokensWithoutKv;
    logicalPosition += retained;
    output.insert(output.end(), result.outputTokens.begin(), result.outputTokens.end());
    retainedCounts.push_back(retained);
    acceptedCounts.push_back(result.acceptedDraftTokens);
    draftedCounts.push_back(result.draftedTokens);
    const auto telemetry = executor.telemetry();
    cycleGpuSeconds.push_back(telemetry.lastDecodeGpuSeconds);
    cycleWallSeconds.push_back(std::chrono::duration<double>(Clock::now() - cycleStarted).count());
    peakAllocated = std::max(peakAllocated, backend.memoryStats().allocatedBytes);
    peakFootprint = std::max(peakFootprint, footprintBytes());
    finished = result.finished;
    if (output.size() > request.maxNewTokens) throw std::runtime_error("decode exceeded its token budget");
  }
  const auto ended = Clock::now();
  const auto telemetry = executor.telemetry();
  const auto optimization = optimizationTelemetry(executor);
  const auto runtimeMemory = executor.actualRuntimeMemory();
  const auto backendMemory = backend.memoryStats();
  const auto replay = replayTelemetry(backend);
  const auto host = engine::queryHostMemory();
  const uint64_t swapAfter = swapBytes();
  std::ostringstream record;
  record << std::setprecision(12) << "{\"type\":\"result\",\"case\":" << json::quote(item.name)
         << ",\"mode\":" << json::quote(mode) << ",\"trial\":" << trial << ",\"order\":" << order
         << ",\"warmup\":" << (warmup ? "true" : "false") << ",\"prompt_tokens\":" << item.prompt.size()
         << ",\"maximum_new_tokens\":" << request.maxNewTokens << ",\"temperature\":" << item.temperature
         << ",\"seed\":" << item.seed << ",\"finished\":" << (finished ? "true" : "false")
         << ",\"output_tokens\":" << output.size() << ",\"output_ids\":";
  array(record, std::span<const uint32_t>(output));
  record << ",\"retained_counts\":"; array(record, std::span<const uint32_t>(retainedCounts));
  record << ",\"accepted_counts\":"; array(record, std::span<const uint32_t>(acceptedCounts));
  record << ",\"drafted_counts\":"; array(record, std::span<const uint32_t>(draftedCounts));
  record << ",\"cycle_gpu_seconds\":"; array(record, std::span<const double>(cycleGpuSeconds));
  record << ",\"cycle_wall_seconds\":"; array(record, std::span<const double>(cycleWallSeconds));
  const double decodeSeconds = std::chrono::duration<double>(ended - prefilled).count();
  record << ",\"setup_seconds\":" << setupSeconds
         << ",\"wall_seconds\":" << std::chrono::duration<double>(ended - started).count()
         << ",\"prefill_wall_seconds\":" << std::chrono::duration<double>(prefilled - started).count()
         << ",\"decode_wall_seconds\":" << decodeSeconds
         << ",\"prefill_gpu_seconds\":" << telemetry.totalPrefillGpuSeconds
         << ",\"decode_gpu_seconds\":" << telemetry.totalDecodeGpuSeconds
         << ",\"decode_tokens_per_second\":" << (decodeSeconds > 0 ? output.size() / decodeSeconds : 0)
         << ",\"decode_commands\":" << metal::BackendInstrumentation::submittedCommands(backend) - commandsBefore
         << ",\"lookup_cycles\":" << optimization.lookupCycles
         << ",\"lookup_proposals\":" << optimization.lookupProposals
         << ",\"lookup_accepted\":" << optimization.lookupAccepted
         << ",\"neural_draft_cycles\":" << optimization.neuralDraftCycles
         << ",\"target_only_cycles\":" << optimization.targetOnlyCycles
         << ",\"gpu_controlled_cycles\":" << optimization.gpuControlledCycles
         << ",\"grouped_gpu_commands\":" << optimization.groupedGpuCommands
         << ",\"target_row_cycles\":";
  array(record, std::span<const uint64_t>(optimization.targetRowCycles));
  record << ",\"replay_hits\":" << replay.hits - replayBefore.hits << ",\"replay_builds\":" << replay.builds - replayBefore.builds
         << ",\"replay_bypasses\":" << replay.bypasses - replayBefore.bypasses << ",\"replay_cached_bytes\":" << replay.cachedBytes
         << ",\"state_bytes\":" << runtimeMemory.stateActualAllocatedBytes
         << ",\"prefill_arena_bytes\":" << runtimeMemory.sharedPrefillActualAllocatedBytes
         << ",\"decode_arena_bytes\":" << runtimeMemory.sharedDecodeActualAllocatedBytes
         << ",\"metal_allocated_bytes\":" << backendMemory.allocatedBytes
         << ",\"metal_peak_allocated_bytes\":" << peakAllocated
         << ",\"metal_process_peak_bytes\":" << backendMemory.peakAllocatedBytes
         << ",\"device_allocated_bytes\":" << backendMemory.deviceCurrentAllocatedBytes
         << ",\"physical_footprint_peak_bytes\":" << peakFootprint
         << ",\"host_available_bytes\":" << (host ? host->available : 0)
         << ",\"swap_before_bytes\":" << swapBefore << ",\"swap_after_bytes\":" << swapAfter << '}';
  executor.end(1);
  clearReplay(backend);
  memoryGate(backend);
  return record.str();
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 5) throw std::invalid_argument("usage: runtime-optimization METALLIB MODEL_ROOT CASES_FILE OUTPUT_JSONL [--modes LIST] [--trials N] [--warmup-tokens N]");
      auto cases = readCases(argv[3]);
      auto modes = splitModes("baseline,lookup,rows1,rows2,rows4,auto,control,replay,all");
      uint32_t trials = 2, warmupTokens = 32;
      for (int index = 5; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view option = argv[index];
        if (option == "--modes") modes = splitModes(argv[index + 1]);
        else if (option == "--trials") trials = count(argv[index + 1]);
        else if (option == "--warmup-tokens") warmupTokens = count(argv[index + 1]);
        else throw std::invalid_argument("unknown benchmark option");
      }
#ifdef SPLASH_BENCH_BASELINE_COMPAT
      if (modes.size() != 1 || modes.front() != "baseline")
        throw std::invalid_argument("the unmodified baseline build accepts --modes baseline only");
#endif
      if (trials > 20 || warmupTokens > 256) throw std::invalid_argument("benchmark trial or warmup limit exceeded");
      std::ofstream output(argv[4], std::ios::app);
      if (!output) throw std::runtime_error("cannot open output JSONL");
      metal::MetalBackend backend(argv[1]);
      memoryGate(backend);
      auto previousHostCheck = Clock::now();
      backend.setOperationGuard([&] {
        if (const auto stats = backend.memoryStats();
            std::max(stats.allocatedBytes, stats.deviceCurrentAllocatedBytes) > kAllocationLimit)
          throw std::runtime_error("benchmark exceeded its 32 GiB Metal allocation limit");
        if (Clock::now() - previousHostCheck > std::chrono::seconds(1)) {
          memoryGate(backend);
          previousHostCheck = Clock::now();
        }
      });
      const std::filesystem::path root = argv[2];
      const auto descriptor = model::inspectModelRoot(root);
      const uint64_t weightBytes = model::modelWeightBytes(root, descriptor);
      memoryGate(backend, weightBytes);
      auto loaded = model::loadModel(backend, root, descriptor);
      ops::ExecutionPlans operators(backend.capabilities());
      uint64_t maximumTokens = 0;
      for (const auto &item : cases) {
        maximumTokens = std::max(maximumTokens, uint64_t{item.prompt.size()} + item.maximumNewTokens + 16);
        if (maximumTokens > loaded.maximumContextTokens()) throw std::invalid_argument("benchmark exceeds model context limit");
        for (const uint32_t token : item.prompt)
          if (token >= loaded.draft.layout.vocabularySize) throw std::invalid_argument("prompt contains token outside vocabulary");
      }
      const auto layout = loaded.targetKvLayout(kv::Format::Int8);
      const uint32_t extentPages = layout.maximumExtentPages();
      const uint32_t wantedPages = (maximumTokens + kv::kPageTokens - 1) / kv::kPageTokens;
      const uint32_t pageCount = (wantedPages + extentPages - 1) / extentPages * extentPages;
      const auto runtimePlan = model::plannedRuntimeMemory(loaded, operators, kv::Format::Int8, ops::GdnStateFormat::Float32);
      memoryGate(backend, uint64_t{pageCount} * layout.bytesPerModelPage() + runtimePlan.laneStatePlannedAllocatedBytes +
                           runtimePlan.sharedPrefillPlannedAllocatedBytes + runtimePlan.sharedDecodePlannedAllocatedBytes);
      const uint64_t reserve = backend.capabilities().physicalMemoryBytes * 15 / 100;
      engine::MemoryGovernor governor(backend, kAllocationLimit, reserve, reserve, engine::queryHostMemory, 0);
      governor.setServing(true);
      kv::PageStorage pages(backend, governor.allocationAdmission(), layout, pageCount, extentPages);
      for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent)
        if (!pages.allocateExtent(extent)) throw std::runtime_error("benchmark KV allocation refused");
      output << "{\"type\":\"metadata\",\"device\":" << json::quote(backend.capabilities().deviceName)
             << ",\"model\":" << json::quote(loaded.name()) << ",\"model_manifest\":" << json::quote(loaded.manifestFingerprintSha256)
#ifdef SPLASH_BENCH_BASELINE_COMPAT
             << ",\"baseline_compat\":true"
#else
             << ",\"baseline_compat\":false"
#endif
             << ",\"weight_bytes\":" << weightBytes << ",\"allocation_limit_bytes\":" << kAllocationLimit
             << ",\"host_reserve_bytes\":" << reserve << ",\"kv_format\":\"int8\",\"state_format\":\"fp32\"}\n" << std::flush;
      for (uint32_t arm = 0; arm < modes.size(); ++arm) {
        const auto result = run(backend, loaded, operators, pages, governor, cases.front(), modes[arm], 0, arm, true, warmupTokens);
        output << result << '\n' << std::flush;
        std::cout << "warmup " << modes[arm] << " complete\n" << std::flush;
      }
      for (uint32_t trial = 0; trial < trials; ++trial)
        for (const auto &item : cases)
          for (uint32_t arm = 0; arm < modes.size(); ++arm) {
            const uint32_t index = trial % 2 ? static_cast<uint32_t>(modes.size()) - 1 - arm : arm;
            output << run(backend, loaded, operators, pages, governor, item, modes[index], trial + 1, arm, false, warmupTokens)
                   << '\n' << std::flush;
            std::cout << "trial " << trial + 1 << ' ' << item.name << ' ' << modes[index] << " complete\n" << std::flush;
          }
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "runtime-optimization: " << error.what() << '\n';
      return 1;
    }
  }
}
