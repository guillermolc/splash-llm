#include "metal/MetalBackend.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace splash::metal;

double median(std::vector<double> values) {
    std::ranges::sort(values);
    const size_t middle = values.size() / 2;
    return (values[middle - 1] + values[middle]) / 2;
}

struct Measurements {
    std::vector<double> host, wall, gpu;
};

void printMeasurements(const Measurements &values) {
    std::cout << "{\"host_ms\":" << median(values.host)
              << ",\"wall_ms\":" << median(values.wall)
              << ",\"gpu_ms\":" << median(values.gpu) << '}';
}

void run(const char *metallib) {
    MetalBackend backend(metallib);
    auto values = backend.allocateBuffer(64 * sizeof(uint32_t), BufferStorage::Shared,
                                          "command-replay-benchmark");
    std::fill_n(static_cast<uint32_t *>(values.contents()), 64, 0);
    const uint32_t count = 64, increment = 1;
    const ComputeDispatch add{"test_add_u32", {{0, values}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {64, 1, 1}};
    const std::vector<ComputeDispatch> graph(256, add);
    Measurements regular, replay;
    uint32_t expected = 0;
    for (const bool enabled : {false, true, true, false}) {
        backend.setCommandReplayEnabled(enabled);
        for (size_t warmup = 0; warmup < 3; ++warmup) {
            (void)backend.submitCommandAsync(graph).wait();
            expected += 256;
        }
        Measurements &measurements = enabled ? replay : regular;
        for (size_t iteration = 0; iteration < 24; ++iteration) {
            const auto start = Clock::now();
            auto ticket = backend.submitCommandAsync(graph);
            const auto submitted = Clock::now();
            const auto timing = ticket.wait();
            const auto end = Clock::now();
            measurements.host.push_back(
                std::chrono::duration<double, std::milli>(submitted - start).count());
            measurements.wall.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
            measurements.gpu.push_back(1000 * timing.gpuSeconds);
            expected += 256;
        }
    }
    for (size_t index = 0; index < 64; ++index)
        if (static_cast<uint32_t *>(values.contents())[index] != expected)
            throw std::runtime_error("replay benchmark produced incorrect values");
    const auto stats = backend.commandReplayStats();
    if (stats.builds != 1 || stats.hits != 52)
        throw std::runtime_error("benchmark did not exercise cached ICB replay");
    std::cout << std::fixed << std::setprecision(6)
              << "{\"benchmark\":\"metal_command_replay\",\"dispatches\":256,"
                 "\"words\":64,\"order\":\"ABBA\",\"samples_per_mode\":48,\"regular\":";
    printMeasurements(regular);
    std::cout << ",\"replay\":";
    printMeasurements(replay);
    std::cout << ",\"hits\":" << stats.hits << ",\"builds\":" << stats.builds
              << ",\"exact_result\":true}\n";
}
} // namespace

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            std::cerr << "usage: metal-command-replay-benchmark <metal-backend.metallib>\n";
            return 2;
        }
        try {
            run(argv[1]);
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
