#pragma once

#include "model/Runtime.hpp"

#include <cstdint>
#include <span>

namespace splash::model {

// Development-only access to completed decode scratch. Only Runtime.cpp
// compiled with SPLASH_RUNTIME_INSTRUMENTATION defines these methods.
// Capture ideas: incoai/splash #550 (ef0e95d4) and #556 (b4006b12).
class RuntimeInstrumentation final {
public:
  struct DraftSelection final {
    std::span<const uint32_t> candidates, proposed, targetTokens;
    std::span<const float> unary, edges;
    uint32_t accepted = 0, retained = 0;
  };

  // Valid until the next runtime command. Position zero's edge table has
  // only predecessor row zero; its other predecessor rows are unwritten.
  [[nodiscard]] static DraftSelection
  lastDraftSelection(const Runtime &runtime, uint32_t lane, uint32_t width);
  // Raw BF16 bit patterns, including the anchor's normalized hidden row.
  [[nodiscard]] static std::span<const uint16_t>
  lastDraftHidden(const Runtime &runtime, uint32_t lane);
};

} // namespace splash::model
