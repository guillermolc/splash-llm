#include "metal/abi/DecodeControl.h"

using namespace metal;

kernel void decode_control_prepare(
    device const DecodeControlState *states [[buffer(0)]],
    device uint *draft_input [[buffer(1)]],
    device uint *target_positions [[buffer(2)]],
    device uint *draft_positions [[buffer(3)]],
    constant DecodeControlParams &params [[buffer(4)]],
    uint lane [[threadgroup_position_in_grid]]) {
  const DecodeControlState state = states[lane];
  const bool active = state.status == 0;
  for (uint row = 0; row < SPLASH_TARGET_VERIFY_ROWS; ++row) {
    const uint index = lane * SPLASH_TARGET_VERIFY_ROWS + row;
    const uint position = active ? state.logical_position + row : 0;
    const uint target_position =
        active ? uint(long(position) + state.target_rope_offset) : 0;
    draft_input[index] = active && row == 0 ? state.anchor : params.mask_token;
    draft_positions[index] = position;
    for (uint axis = 0; axis < 3; ++axis)
      target_positions[index * 3 + axis] = target_position;
  }
}

kernel void decode_control_advance(
    device DecodeControlState *states [[buffer(0)]],
    device const uint *output [[buffer(1)]],
    device const uint *retained_counts [[buffer(2)]],
    device const uint *accepted_counts [[buffer(3)]],
    device uint *emitted [[buffer(4)]],
    device DecodeControlState *completed_states [[buffer(5)]],
    constant DecodeControlParams &params [[buffer(6)]],
    uint lane [[threadgroup_position_in_grid]]) {
  device DecodeControlState &state = states[lane];
  state.emitted_tokens = 0;
  if (state.status != 0) {
    completed_states[lane] = state;
    return;
  }

  const uint retained = retained_counts[lane];
  const uint accepted = accepted_counts[lane];
  if (state.generated_tokens >= state.maximum_new_tokens ||
      retained == 0 || retained > params.target_rows ||
      retained > state.maximum_new_tokens - state.generated_tokens ||
      accepted >= params.target_rows || retained > accepted + 1) {
    state.status = SPLASH_DECODE_INVALID_COUNT;
    completed_states[lane] = state;
    return;
  }
  output += lane * SPLASH_TARGET_VERIFY_ROWS;
  // Every retained token is checked, including the next pending anchor.
  if (state.anchor >= params.vocabulary) {
    state.status = SPLASH_DECODE_INVALID_TOKEN;
    completed_states[lane] = state;
    return;
  }
  for (uint row = 0; row < retained; ++row) {
    if (output[row] >= params.vocabulary) {
      state.status = SPLASH_DECODE_INVALID_TOKEN;
      completed_states[lane] = state;
      return;
    }
  }
  const ulong next_position = ulong(state.logical_position) + retained;
  const long next_rope = long(next_position) + state.target_rope_offset;
  if (next_position > SPLASH_MAXIMUM_CONTEXT_TOKENS || next_rope < 0 ||
      next_rope + SPLASH_TARGET_VERIFY_ROWS - 1 > 0xffffffffl) {
    state.status = SPLASH_DECODE_POSITION_OVERFLOW;
    completed_states[lane] = state;
    return;
  }

  emitted += lane * SPLASH_DECODE_EMISSION_ROWS;
  emitted[0] = state.anchor;
  for (uint row = 0; row + 1 < retained; ++row)
    emitted[row + 1] = output[row];
  const uint next_anchor = output[retained - 1];
  state.anchor = next_anchor;
  state.logical_position = uint(next_position);
  state.generated_tokens += retained;
  state.last_retained = retained;
  state.last_accepted = accepted;
  state.emitted_tokens = retained;
  ++state.cycles;

  // Match Runtime::emitTerminalAnchor: a selected terminal anchor needs no
  // target KV row, and is emitted as soon as the cycle selects it.
  if (state.generated_tokens < state.maximum_new_tokens) {
    const bool stopped = next_anchor == params.stop_token_0 ||
                         next_anchor == params.stop_token_1;
    if (stopped || state.maximum_new_tokens - state.generated_tokens == 1) {
      emitted[state.emitted_tokens++] = next_anchor;
      ++state.generated_tokens;
      if (stopped)
        state.status |= SPLASH_DECODE_STOPPED;
    }
  }
  if (state.generated_tokens == state.maximum_new_tokens)
    state.status |= SPLASH_DECODE_BUDGET_EXHAUSTED;
  state.maximum_retained = state.status == 0
                               ? min(params.target_rows,
                                     state.maximum_new_tokens - state.generated_tokens)
                               : 0;
  completed_states[lane] = state;
}

kernel void decode_control_patch(
    device const DecodeControlState *states [[buffer(0)]],
    device const DecodeControlPatch *patches [[buffer(1)]],
    device uint *destination [[buffer(2)]],
    constant DecodeControlPatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.count)
    return;
  const DecodeControlPatch patch = patches[index];
  const DecodeControlState state = states[patch.lane];
  uint source = 0;
  switch (patch.source) {
  case SPLASH_DECODE_PATCH_POSITION: source = state.logical_position; break;
  case SPLASH_DECODE_PATCH_PREVIOUS_POSITION:
    source = state.logical_position - state.last_retained; break;
  case SPLASH_DECODE_PATCH_ANCHOR: source = state.anchor; break;
  case SPLASH_DECODE_PATCH_GENERATED: source = state.generated_tokens; break;
  case SPLASH_DECODE_PATCH_REMAINING:
    source = state.maximum_new_tokens - state.generated_tokens; break;
  case SPLASH_DECODE_PATCH_MAXIMUM_RETAINED: source = state.maximum_retained; break;
  case SPLASH_DECODE_PATCH_RETAINED: source = state.last_retained; break;
  case SPLASH_DECODE_PATCH_ACCEPTED: source = state.last_accepted; break;
  case SPLASH_DECODE_PATCH_ACTIVE: source = state.status == 0 ? 1 : 0; break;
  case SPLASH_DECODE_PATCH_CYCLES: source = state.cycles; break;
  }
  // Unsigned multiplication avoids overflow even for UINT_MAX operands;
  // apply a signed addend without converting a large product to signed.
  const ulong product = ulong(source) * patch.multiplier;
  const ulong value = patch.addend < 0
                          ? (product > ulong(-long(patch.addend))
                                 ? product - ulong(-long(patch.addend)) : 0)
                          : min(product, 0xfffffffffffffffful - uint(patch.addend)) +
                                uint(patch.addend);
  destination[patch.destination_word] = uint(min(value, ulong(patch.limit)));
}
