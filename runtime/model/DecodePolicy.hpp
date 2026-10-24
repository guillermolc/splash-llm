#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace splash::model {

struct RuntimeOptimizationOptions final {
  bool promptLookup = false;
  bool gpuControl = false;
  bool commandReplay = false;
  bool adaptiveRows = false;
  uint32_t verifyRows = 8;

  static bool enabled(const char *name, bool fallback) {
    const char *value = std::getenv(name);
    if (!value)
      return fallback;
    const std::string_view text(value);
    if (text == "1" || text == "true")
      return true;
    if (text == "0" || text == "false")
      return false;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
  }

  static RuntimeOptimizationOptions fromEnvironment() {
    RuntimeOptimizationOptions result;
    const bool all = enabled("SPLASH_RUNTIME_OPTIMIZATIONS", false);
    result.promptLookup = enabled("SPLASH_PROMPT_LOOKUP", all);
    result.gpuControl = enabled("SPLASH_GPU_DECODE_CONTROL", all);
    result.commandReplay = enabled("SPLASH_COMMAND_REPLAY", all);
    if (const char *value = std::getenv("SPLASH_VERIFY_ROWS")) {
      const std::string_view rows(value);
      if (rows == "auto")
        result.adaptiveRows = true;
      else if (rows == "1" || rows == "2" || rows == "4" || rows == "8")
        result.verifyRows = static_cast<uint32_t>(rows[0] - '0');
      else
        throw std::invalid_argument("SPLASH_VERIFY_ROWS must be auto, 1, 2, 4 or 8");
    } else {
      result.adaptiveRows = all;
    }
    return result;
  }
};

// Bounded, token-exact prompt lookup. The hash only finds candidates: every
// match is checked against the stored token IDs before producing proposals.
// The pending anchor belongs to the suffix being matched but is not appended
// until its target KV row is committed. This keeps prefix-cache resumes and
// the first decode cycle consistent with normal generation.
class PromptLookup final {
public:
  explicit PromptLookup(uint32_t minimumMatch = 16,
                        uint32_t maximumIndexedSuffixes = 65536,
                        uint32_t maximumHistory = 262144)
      : minimumMatch_(minimumMatch), maximumIndexed_(maximumIndexedSuffixes),
        maximumHistory_(maximumHistory) {
    if (minimumMatch < 2 || minimumMatch > 256 || !maximumIndexedSuffixes ||
        maximumHistory < minimumMatch)
      throw std::invalid_argument("invalid prompt lookup limits");
  }

  void reset(std::span<const uint32_t> tokens) {
    history_.clear();
    positions_.clear();
    order_.clear();
    disabled_ = false;
    append(tokens);
  }

  void append(std::span<const uint32_t> tokens) {
    if (disabled_)
      return;
    if (tokens.size() > maximumHistory_ - history_.size()) {
      // A resource limit changes only drafting, never target semantics.
      history_.clear();
      positions_.clear();
      order_.clear();
      disabled_ = true;
      return;
    }
    for (uint32_t token : tokens) {
      history_.push_back(token);
      if (history_.size() < minimumMatch_)
        continue;
      const uint32_t begin = static_cast<uint32_t>(history_.size()) - minimumMatch_;
      const uint64_t hash = hashTokens(std::span(history_).subspan(begin, minimumMatch_));
      Positions &positions = positions_[hash];
      if (positions.count == positions.values.size()) {
        std::move(positions.values.begin() + 1, positions.values.end(), positions.values.begin());
        --positions.count;
      }
      positions.values[positions.count++] = begin;
      order_.push_back({hash, begin});
      while (order_.size() > maximumIndexed_) {
        const auto old = order_.front();
        order_.pop_front();
        auto found = positions_.find(old.hash);
        if (found == positions_.end())
          continue;
        Positions &kept = found->second;
        auto end = std::remove(kept.values.begin(), kept.values.begin() + kept.count, old.begin);
        kept.count = static_cast<uint32_t>(end - kept.values.begin());
        if (!kept.count)
          positions_.erase(found);
      }
    }
  }

  [[nodiscard]] std::vector<uint32_t> propose(uint32_t anchor,
                                             uint32_t maximum = 7) const {
    if (!maximum || disabled_ || history_.size() + 1 < minimumMatch_)
      return {};
    const uint32_t suffix = static_cast<uint32_t>(history_.size()) - (minimumMatch_ - 1);
    uint64_t hash = hashTokens(std::span(history_).subspan(suffix));
    hash = hashToken(hash, anchor);
    const auto found = positions_.find(hash);
    if (found == positions_.end())
      return {};
    uint32_t bestBegin = 0, bestCount = 0, bestMatched = 0;
    bool ambiguous = false;
    for (uint32_t i = found->second.count; i > 0; --i) {
      const uint32_t begin = found->second.values[i - 1];
      const uint32_t after = begin + minimumMatch_;
      if (after >= history_.size() || history_[after - 1] != anchor ||
          !std::equal(history_.begin() + suffix, history_.end(), history_.begin() + begin))
        continue;
      const uint32_t count = std::min<uint32_t>(maximum,
          static_cast<uint32_t>(history_.size()) - after);
      // Identical function bodies can lead to different next functions.
      // Extend the suffix backwards to distinguish their surrounding code,
      // rather than choosing the latest hash hit. The bounded extension is
      // inexpensive compared with a target pass and avoids low-yield copies.
      uint32_t matched = minimumMatch_;
      while (matched < 512 && begin >= matched - minimumMatch_ + 1 &&
             suffix >= matched - minimumMatch_ + 1 &&
             history_[begin - (matched - minimumMatch_) - 1] ==
                 history_[suffix - (matched - minimumMatch_) - 1])
        ++matched;
      if (matched > bestMatched || (matched == bestMatched && count > bestCount)) {
        bestCount = count;
        bestBegin = after;
        bestMatched = matched;
        ambiguous = false;
      } else if (matched == bestMatched && count == bestCount &&
                 !std::equal(history_.begin() + after, history_.begin() + after + count,
                             history_.begin() + bestBegin)) {
        ambiguous = true;
      }
    }
    if (ambiguous)
      return {};
    return {history_.begin() + bestBegin, history_.begin() + bestBegin + bestCount};
  }

  [[nodiscard]] size_t indexedSuffixes() const noexcept { return order_.size(); }
  [[nodiscard]] size_t historyTokens() const noexcept { return history_.size(); }

private:
  static uint64_t hashToken(uint64_t hash, uint32_t token) noexcept {
    return (hash ^ token) * 1099511628211ULL;
  }
  static uint64_t hashTokens(std::span<const uint32_t> tokens) noexcept {
    uint64_t hash = 14695981039346656037ULL;
    for (uint32_t token : tokens)
      hash = hashToken(hash, token);
    return hash;
  }
  struct Positions { std::array<uint32_t, 16> values{}; uint32_t count = 0; };
  struct Indexed { uint64_t hash; uint32_t begin; };
  uint32_t minimumMatch_, maximumIndexed_, maximumHistory_;
  bool disabled_ = false;
  std::vector<uint32_t> history_;
  std::unordered_map<uint64_t, Positions> positions_;
  std::deque<Indexed> order_;
};

// The controller prices complete measured rounds, including host/driver work.
// It never infers a twofold saving from halving the row count: the quantized
// matrix kernels may still use an eight-row tile. Lookup rounds have a
// different cost and must not contaminate neural-draft prices.
//
// For M5 dense M8 kernels, callers may preserve eight rows while two or more
// recent eight-row rounds average at least six accepted proposals. This is a
// conservative way to avoid expensive narrow exploration on short, strongly
// accepted answers: it can miss a faster narrow configuration, so it is not
// an optimality claim. We invent no narrower-width timing. Once wide acceptance
// falls, the ordinary measured-cost exploration resumes, four rows first.
class VerifyRowController final {
public:
  static constexpr std::array<uint32_t, 4> widths{8, 4, 2, 1};

  VerifyRowController() = default;
  explicit VerifyRowController(bool preserveWideWithStrongAcceptance)
      : preserveWideWithStrongAcceptance_(preserveWideWithStrongAcceptance) {}

  [[nodiscard]] uint32_t choose(uint64_t contextTokens) const noexcept {
    // Recalibrate after crossing a context bucket; do not extrapolate the
    // attention slope of a short prompt to an agent's long context.
    const uint64_t bucket = contextTokens / 4096;
    if (bucket != bucket_)
      return widths[0];
    if (samples_[0].count < 2 ||
        (preserveWideWithStrongAcceptance_ && wideCount_ >= 2 &&
         wideAcceptedSum_ >= 6 * wideCount_))
      return widths[0];
    for (size_t i = 0; i < widths.size(); ++i)
      if (samples_[i].count < 2)
        return widths[i];
    if (rounds_ && rounds_ % 16 == 0)
      return widths[(rounds_ / 16) % widths.size()];
    uint32_t best = current_;
    double bestRate = rate(best);
    for (uint32_t rows : widths) {
      const double candidate = rate(rows);
      if (candidate > bestRate * 1.05) {
        best = rows;
        bestRate = candidate;
      }
    }
    return best;
  }

  void observe(uint32_t rows, uint32_t accepted, double milliseconds,
               uint64_t contextTokens) {
    if (!valid(rows) || accepted >= rows || !std::isfinite(milliseconds) || milliseconds <= 0)
      throw std::invalid_argument("invalid verification timing sample");
    const uint64_t bucket = contextTokens / 4096;
    if (bucket != bucket_) {
      samples_ = {};
      acceptance_ = {};
      acceptanceCount_ = {};
      wideAccepted_ = {};
      wideCount_ = wideNext_ = wideAcceptedSum_ = 0;
      rounds_ = 0;
      bucket_ = bucket;
      current_ = 8;
    }
    Sample &sample = samples_[index(rows)];
    sample.milliseconds = sample.count ? 0.8 * sample.milliseconds + 0.2 * milliseconds : milliseconds;
    ++sample.count;
    if (rows == widths[0]) {
      if (wideCount_ == wideAccepted_.size())
        wideAcceptedSum_ -= wideAccepted_[wideNext_];
      else
        ++wideCount_;
      wideAccepted_[wideNext_] = accepted;
      wideAcceptedSum_ += accepted;
      wideNext_ = (wideNext_ + 1) % wideAccepted_.size();
    }
    for (uint32_t i = 0; i < rows - 1; ++i) {
      if (i > accepted)
        break;
      const double value = i < accepted ? 1.0 : 0.0;
      ++acceptanceCount_[i];
      const double alpha = std::max(0.05, 1.0 / acceptanceCount_[i]);
      acceptance_[i] += alpha * (value - acceptance_[i]);
    }
    current_ = rows;
    ++rounds_;
  }

  [[nodiscard]] static bool valid(uint32_t rows) noexcept {
    return rows == 1 || rows == 2 || rows == 4 || rows == 8;
  }

private:
  static size_t index(uint32_t rows) noexcept {
    for (size_t i = 0; i < widths.size(); ++i)
      if (widths[i] == rows)
        return i;
    return 0;
  }
  [[nodiscard]] double rate(uint32_t rows) const noexcept {
    const Sample &sample = samples_[index(rows)];
    if (!sample.count)
      return 0;
    double expected = 1, survival = 1;
    for (uint32_t i = 0; i < rows - 1; ++i) {
      survival *= acceptance_[i];
      expected += survival;
    }
    return expected / sample.milliseconds;
  }
  struct Sample { uint32_t count = 0; double milliseconds = 0; };
  std::array<Sample, 4> samples_{};
  // Conditional acceptance is populated only by actual observations. A
  // never-reached proposal contributes nothing until its prefix is observed.
  std::array<double, 7> acceptance_{};
  std::array<uint32_t, 7> acceptanceCount_{};
  std::array<uint32_t, 8> wideAccepted_{};
  uint32_t wideCount_ = 0, wideNext_ = 0, wideAcceptedSum_ = 0;
  uint64_t bucket_ = std::numeric_limits<uint64_t>::max();
  uint64_t rounds_ = 0;
  uint32_t current_ = 8;
  bool preserveWideWithStrongAcceptance_ = false;
};

} // namespace splash::model
