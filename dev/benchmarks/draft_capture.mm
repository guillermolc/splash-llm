// Single-lane greedy DFlash capture for deployment-matched adaptation.
// Based on the completed-scratch capture in incoai/splash #550 (ef0e95d4)
// and final-hidden readback in #556 (b4006b12). This tool does no profiling
// replay: token labels stop at retained rows, before any incorrect prefix.

#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "ops/PageStorage.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <sys/resource.h>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr uint64_t kMemoryLimit = 24ULL << 30;
constexpr uint32_t kMaximumPromptTokens = 8192;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
static_assert(kPositions == 7 && kCandidates == 16 && kRows == 8);

struct Options final {
  std::filesystem::path model, metallib, prompt, output;
  uint32_t cycles = 32, maxTokens = 256;
  bool hidden = false;
};

void usage(std::ostream &out) {
  out << "usage: draft-capture --model ABS --metallib PATH --prompt PATH.u32 "
         "--output PATH.jsonl [--cycles 1..256] [--max-tokens 1..2048] "
         "[--hidden yes|no]\n";
}

uint32_t count(std::string_view value, std::string_view option, uint32_t maximum) {
  uint32_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      !result || result > maximum)
    throw std::invalid_argument(std::string(option) + " must be from 1 to " +
                                std::to_string(maximum));
  return result;
}

Options parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    const std::string_view option(argv[index]);
    if (index + 1 == argc)
      throw std::invalid_argument(std::string(option) + " requires a value");
    const char *value = argv[index + 1];
    if (option == "--model") options.model = value;
    else if (option == "--metallib") options.metallib = value;
    else if (option == "--prompt") options.prompt = value;
    else if (option == "--output") options.output = value;
    else if (option == "--cycles") options.cycles = count(value, option, 256);
    else if (option == "--max-tokens") options.maxTokens = count(value, option, 2048);
    else if (option == "--hidden") {
      const std::string_view selected(value);
      if (selected != "yes" && selected != "no")
        throw std::invalid_argument("--hidden takes yes or no");
      options.hidden = selected == "yes";
    } else throw std::invalid_argument("unknown option: " + std::string(option));
  }
  if (options.model.empty() || options.metallib.empty() || options.prompt.empty() ||
      options.output.empty())
    throw std::invalid_argument("--model, --metallib, --prompt and --output are required");
  if (!options.model.is_absolute())
    throw std::invalid_argument("--model must be an absolute path");
  if (!std::filesystem::is_directory(options.model))
    throw std::invalid_argument("--model must name an installed model directory");
  if (!std::filesystem::is_regular_file(options.metallib))
    throw std::invalid_argument("--metallib must name an existing file");
  if (std::filesystem::exists(options.output))
    throw std::invalid_argument("--output already exists; select a new capture path");
  return options;
}

std::vector<uint32_t> readPrompt(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open prompt: " + path.string());
  const std::streamoff bytes = input.tellg();
  if (bytes <= 0 || bytes % 4 || bytes / 4 > kMaximumPromptTokens)
    throw std::invalid_argument("prompt must contain 1..8192 little-endian uint32 token IDs");
  std::vector<uint8_t> encoded(static_cast<size_t>(bytes));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(encoded.data()), bytes);
  if (!input) throw std::runtime_error("could not read the complete prompt");
  std::vector<uint32_t> tokens(encoded.size() / 4);
  for (size_t i = 0; i < tokens.size(); ++i)
    for (uint32_t byte = 0; byte < 4; ++byte)
      tokens[i] |= uint32_t{encoded[i * 4 + byte]} << (byte * 8);
  return tokens;
}

std::string promptSha256(std::span<const uint32_t> tokens) {
  std::vector<uint8_t> bytes(tokens.size() * 4);
  for (size_t index = 0; index < tokens.size(); ++index)
    for (uint32_t byte = 0; byte < 4; ++byte)
      bytes[index * 4 + byte] = static_cast<uint8_t>(tokens[index] >> (byte * 8));
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (unsigned char byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 15]);
  }
  return result;
}

template <class T>
void array(std::ostream &out, std::span<const T> values) {
  out << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if constexpr (std::is_floating_point_v<T>)
      if (!std::isfinite(values[index]))
        throw std::runtime_error("capture contains a nonfinite score");
    if (index) out << ',';
    out << values[index];
  }
  out << ']';
}

template <class T>
void matrix(std::ostream &out, std::span<const T> values, uint32_t rows,
            uint32_t columns) {
  if (values.size() != uint64_t{rows} * columns)
    throw std::runtime_error("invalid capture matrix shape");
  out << '[';
  for (uint32_t row = 0; row < rows; ++row) {
    if (row) out << ',';
    array(out, values.subspan(uint64_t{row} * columns, columns));
  }
  out << ']';
}

void edges(std::ostream &out, std::span<const float> values) {
  constexpr std::array<float, kCandidates> unwritten{};
  out << '[';
  for (uint32_t position = 0; position < kPositions; ++position) {
    if (position) out << ',';
    out << '[';
    for (uint32_t predecessor = 0; predecessor < kCandidates; ++predecessor) {
      if (predecessor) out << ',';
      // The position-zero kernel writes only predecessor zero. Never read
      // its other rows, which may contain stale or uninitialized scratch.
      if (!position && predecessor)
        array(out, std::span<const float>(unwritten));
      else
        array(out, values.subspan((position * kCandidates + predecessor) *
                                      kCandidates, kCandidates));
    }
    out << ']';
  }
  out << ']';
}

ModelStepResult prefill(model::Runtime &runtime, const EngineRequest &request,
                        std::span<const uint32_t> pageIds) {
  runtime.beginColdRequest(request.modelView(), 0);
  ModelStepResult final;
  for (uint32_t offset = 0; offset < request.prompt.size();) {
    const uint32_t rows = std::min<uint32_t>(model::ExecutionLimits::prefillTokenBudget,
                                            request.prompt.size() - offset);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{request.id, rows, offset}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{request.id, offset, rows, pageIds, 1};
    item.inputTokens = std::span(request.prompt).subspan(offset, rows);
    auto results = runtime.prefill(plan, {&item, 1});
    if (results.size() != 1 || results[0].consumedPromptTokens != rows)
      throw std::runtime_error("prefill returned an invalid row count");
    if (!results[0].failure.empty()) throw std::runtime_error(results[0].failure);
    final = std::move(results[0]);
    offset += rows;
  }
  return final;
}

void capture(const Options &options) {
  const std::vector<uint32_t> prompt = readPrompt(options.prompt);
  const auto descriptor = model::inspectModelRoot(options.model);
  if (descriptor.draft.hiddenSize != 5120 || descriptor.draft.selectorRank != 256)
    throw std::invalid_argument("capture requires a 5120-wide draft with selector rank 256");
  if (uint64_t{prompt.size()} + options.maxTokens + kRows >
      descriptor.capabilities.maximumContextTokens)
    throw std::invalid_argument("prompt and generation budget exceed the model context");
  if (std::ranges::any_of(prompt, [&](uint32_t token) {
        return token >= descriptor.draft.vocabularySize;
      }))
    throw std::invalid_argument("prompt token outside the model vocabulary");
  if (model::modelWeightBytes(options.model, descriptor) >= kMemoryLimit)
    throw std::runtime_error("model weights exceed the 24 GiB capture budget");

  metal::MetalBackend backend(options.metallib.string());
  const model::LoadedModel loaded = model::loadModel(backend, options.model, descriptor);
  ops::ExecutionPlans operators(backend.capabilities());
  constexpr auto stateFormat = ops::GdnStateFormat::Float32;
  const kv::Layout layout = loaded.targetKvLayout(kv::Format::Int8);
  const uint32_t pagesNeeded = (prompt.size() + options.maxTokens + kRows +
                                kv::kPageTokens - 1) / kv::kPageTokens + 1;
  const uint32_t extentPages = layout.maximumExtentPages();
  const uint32_t pageCount = (pagesNeeded + extentPages - 1) / extentPages * extentPages;
  const auto plan = model::plannedRuntimeMemory(loaded, operators, kv::Format::Int8, stateFormat);
  const uint64_t plannedBytes = backend.memoryStats().allocatedBytes +
      uint64_t{pageCount} * layout.bytesPerModelPage() + plan.laneStatePlannedAllocatedBytes +
      plan.sharedPrefillPlannedAllocatedBytes + plan.sharedDecodePlannedAllocatedBytes +
      model::kPipelineReserveBytes + model::kRuntimeOverheadReserveBytes;
  if (plannedBytes > kMemoryLimit)
    throw std::runtime_error("model, KV and runtime exceed the 24 GiB capture budget");
  MemoryGovernor governor(backend, kMemoryLimit, 1, 1, queryHostMemory,
                           model::kRuntimeOverheadReserveBytes);
  governor.setServing(true);
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, pageCount, extentPages);
  for (uint32_t extent = 0; extent < pageCount / extentPages; ++extent)
    if (!pages.allocateExtent(extent))
      throw std::runtime_error("could not allocate capture KV within the memory budget");
  model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                 loaded.stateLayout(stateFormat), nullptr);
  model::RuntimeContext context{backend, loaded, pages, states, operators};
  model::Runtime runtime(context);
  std::vector<uint32_t> pageIds(pagesNeeded);
  for (uint32_t index = 0; index < pagesNeeded; ++index) pageIds[index] = index;
  EngineRequest request;
  request.id = 1;
  request.prompt = prompt;
  request.maxNewTokens = options.maxTokens;
  request.sampling.temperature = 0.0F;

  std::ofstream output(options.output);
  if (!output) throw std::runtime_error("cannot create output: " + options.output.string());
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  const auto started = std::chrono::steady_clock::now();
  const auto initial = prefill(runtime, request, pageIds);
  output << "{\"kind\":\"metadata\",\"schema_version\":1,\"model\":"
         << json::quote(options.model.string()) << ",\"prompt_tokens\":" << prompt.size()
         << ",\"prompt_sha256\":" << json::quote(promptSha256(prompt))
         << ",\"hidden_size\":5120,\"rank\":256,\"top_k\":16,\"query_rows\":8"
         << ",\"model_fingerprint\":" << json::quote(loaded.manifestFingerprintSha256)
         << ",\"device\":" << json::quote(backend.capabilities().deviceName)
         << ",\"kv_format\":\"int8\",\"state_format\":\"fp32\",\"sampling\":false"
         << ",\"sampling_temperature\":0,\"thinking\":null,\"thinking_source\":\"provided_token_ids\""
         << ",\"capture_hidden\":" << (options.hidden ? "true" : "false")
         << ",\"cycles_requested\":" << options.cycles << ",\"max_tokens\":" << options.maxTokens
         << ",\"prefill_output_tokens\":";
  array(output, std::span<const uint32_t>(initial.outputTokens));
  output << ",\"prefill_first_token\":";
  if (initial.firstToken) output << *initial.firstToken;
  else output << "null";
  output << "}\n";

  uint64_t position = prompt.size(), emitted = initial.outputTokens.size();
  uint32_t cycleCount = 0;
  bool finished = initial.finished;
  while (cycleCount < options.cycles && !finished && emitted < options.maxTokens) {
    BatchPlan batch{.kind = WorkKind::Decode, .items = {{1, 0, 0}},
                     .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{1, position, 0, pageIds, 1};
    auto results = runtime.decode(batch, {&item, 1});
    if (results.size() != 1) throw std::runtime_error("capture decode width changed");
    const auto &result = results[0];
    if (!result.failure.empty()) throw std::runtime_error(result.failure);
    if (result.outputTokens.empty() || result.outputTokensWithoutKv > result.outputTokens.size())
      throw std::runtime_error("capture decode returned invalid output tokens");
    const auto selected = model::RuntimeInstrumentation::lastDraftSelection(runtime, 0, 1);
    if (!selected.retained || selected.retained > kRows || selected.accepted > kPositions)
      throw std::runtime_error("capture decode returned invalid acceptance counts");
    output << "{\"kind\":\"cycle\",\"request_id\":1,\"cycle\":" << cycleCount
           << ",\"position\":" << position << ",\"anchor\":" << result.outputTokens.front()
           << ",\"candidates\":";
    matrix(output, selected.candidates, kPositions, kCandidates);
    output << ",\"unary\":";
    matrix(output, selected.unary, kPositions, kCandidates);
    output << ",\"proposed\":";
    array(output, selected.proposed);
    output << ",\"edges\":";
    edges(output, selected.edges);
    // Only retained rows have verifier-correct prefixes. Later raw verify
    // rows may follow an incorrect proposal and are deliberately omitted.
    output << ",\"target_tokens\":";
    array(output, selected.targetTokens.first(selected.retained));
    output << ",\"accepted\":" << selected.accepted << ",\"retained\":" << selected.retained
           << ",\"output_tokens\":";
    array(output, std::span<const uint32_t>(result.outputTokens));
    output << ",\"output_tokens_without_kv\":" << result.outputTokensWithoutKv
           << ",\"finished\":" << (result.finished ? "true" : "false");
    if (options.hidden) {
      output << ",\"draft_hidden_bf16\":";
      matrix(output, model::RuntimeInstrumentation::lastDraftHidden(runtime, 0), kRows, 5120);
    }
    output << "}\n";
    position += result.outputTokens.size() - result.outputTokensWithoutKv;
    emitted += result.outputTokens.size();
    finished = result.finished;
    ++cycleCount;
  }
  const auto telemetry = runtime.telemetry();
  const auto memory = backend.memoryStats();
  const auto budget = governor.snapshot();
  rusage process{};
  const bool haveProcessMemory = getrusage(RUSAGE_SELF, &process) == 0;
  output << "{\"kind\":\"summary\",\"cycle_count\":" << cycleCount
         << ",\"output_token_count\":" << emitted
         << ",\"finished\":" << (finished ? "true" : "false")
         << ",\"stop_reason\":" << json::quote(finished ? "finished" :
                            emitted >= options.maxTokens ? "max_tokens" : "cycles")
         << ",\"prefill_gpu_seconds\":" << telemetry.totalPrefillGpuSeconds
         << ",\"decode_gpu_seconds\":" << telemetry.totalDecodeGpuSeconds
         << ",\"wall_seconds\":" << std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - started).count()
         << ",\"allocated_bytes\":" << memory.allocatedBytes
         << ",\"peak_allocated_bytes\":" << memory.peakAllocatedBytes
         << ",\"device_peak_allocated_bytes\":" << memory.devicePeakAllocatedBytes
         << ",\"memory_limit_bytes\":" << budget.limitBytes
         << ",\"charged_bytes\":" << budget.chargedBytes;
  // Darwin reports ru_maxrss in bytes.
  if (haveProcessMemory) output << ",\"process_peak_rss_bytes\":" << process.ru_maxrss;
  output << "}\n";
  output.close();
  runtime.end(1);
  std::cerr << "draft-capture: wrote " << cycleCount << " cycles to " << options.output << '\n';
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        usage(std::cout);
        return 0;
      }
      capture(parse(argc, argv));
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "draft-capture: " << error.what() << '\n';
      return 1;
    }
  }
}
