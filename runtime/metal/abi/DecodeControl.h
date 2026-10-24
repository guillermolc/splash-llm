#pragma once

#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A ready lane has status zero. Terminal bits are cumulative: a stop token
// may also consume the last budgeted output. Invalid cycles emit no tokens
// and leave the anchor, position and output count at their previous values.
#define SPLASH_DECODE_STOPPED 1u
#define SPLASH_DECODE_BUDGET_EXHAUSTED 2u
#define SPLASH_DECODE_INVALID_COUNT 4u
#define SPLASH_DECODE_INVALID_TOKEN 8u
#define SPLASH_DECODE_POSITION_OVERFLOW 16u
#define SPLASH_DECODE_EMISSION_ROWS (SPLASH_TARGET_VERIFY_ROWS + 1u)

// The pending anchor has been selected but has no target KV row yet and is
// not included in generated_tokens. logical_position is the target KV length.
// A text continuation after images may supply its fixed M-RoPE delta.
struct DecodeControlState {
  uint32_t anchor;
  uint32_t logical_position;
  uint32_t generated_tokens;
  uint32_t maximum_new_tokens;
  int32_t target_rope_offset;
  uint32_t status;
  uint32_t last_retained;
  uint32_t last_accepted;
  uint32_t emitted_tokens;
  uint32_t cycles;
  uint32_t maximum_retained;
  uint32_t reserved;
};

static_assert(sizeof(DecodeControlState) == 48,
              "Decode control state is 48 bytes on both sides");

struct DecodeControlParams {
  uint32_t vocabulary;
  uint32_t mask_token;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t target_rows;
};

static_assert(sizeof(DecodeControlParams) == 20,
              "Decode control parameters are 20 bytes on both sides");

// Expressions used to update a packed mutable parameter/grid buffer before
// a repeated graph. Position means the NEXT cycle's position; previous
// position is the just-verified cycle's position (position-last_retained).
#define SPLASH_DECODE_PATCH_POSITION 0u
#define SPLASH_DECODE_PATCH_PREVIOUS_POSITION 1u
#define SPLASH_DECODE_PATCH_ANCHOR 2u
#define SPLASH_DECODE_PATCH_GENERATED 3u
#define SPLASH_DECODE_PATCH_REMAINING 4u
#define SPLASH_DECODE_PATCH_MAXIMUM_RETAINED 5u
#define SPLASH_DECODE_PATCH_RETAINED 6u
#define SPLASH_DECODE_PATCH_ACCEPTED 7u
#define SPLASH_DECODE_PATCH_ACTIVE 8u
#define SPLASH_DECODE_PATCH_CYCLES 9u
#define SPLASH_DECODE_PATCH_SOURCE_COUNT 10u

// Writes one uint32: clamp(source * multiplier + addend, 0, limit).
// destination_word is relative to one packed parameter allocation, never an
// unchecked device address. Descriptors must have unique destinations and
// are validated by DecodeControl::makePatches before reaching the GPU.
struct DecodeControlPatch {
  uint32_t destination_word;
  uint32_t source;
  uint32_t lane;
  uint32_t multiplier;
  int32_t addend;
  uint32_t limit;
};

static_assert(sizeof(DecodeControlPatch) == 24,
              "Decode control patches are 24 bytes on both sides");

struct DecodeControlPatchParams {
  uint32_t count;
};

static_assert(sizeof(DecodeControlPatchParams) == 4,
              "Decode control patch parameters are 4 bytes on both sides");
