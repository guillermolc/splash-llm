#include <metal_stdlib>

using namespace metal;

struct CommandReplayArguments {
    command_buffer commands [[id(0)]];
};

// Matches ReplayGrid in MetalBackend.mm. The address names the original
// MTLDispatchThreadgroupsIndirectArguments, including a buffer view's offset.
struct CommandReplayGrid {
    ulong address;
    uint threadsX;
    uint threadsY;
    uint threadsZ;
    uint reserved;
};

kernel void command_replay_update_grids(
    constant CommandReplayArguments &arguments [[buffer(0)]],
    device const CommandReplayGrid *grids [[buffer(1)]],
    constant uint2 &range [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= range.y) return;
    const uint commandIndex = range.x + index;
    const CommandReplayGrid grid = grids[commandIndex];
    device const uint *counts = reinterpret_cast<device const uint *>(grid.address);
    compute_command command(arguments.commands, commandIndex);
    command.concurrent_dispatch_threadgroups(
        uint3(counts[0], counts[1], counts[2]),
        uint3(grid.threadsX, grid.threadsY, grid.threadsZ));
}
