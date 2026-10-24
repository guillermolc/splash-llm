#include "model/DecodePolicy.hpp"

#include <cstdio>
#include <numeric>

using namespace splash::model;

static void check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

int main() {
  try {
    std::vector<uint32_t> prompt(64);
    std::iota(prompt.begin(), prompt.end(), 0);
    PromptLookup lookup;
    lookup.reset(prompt);
    check(lookup.propose(1000).empty(), "unseen suffix must not draft");
    const std::vector<uint32_t> repeated(prompt.begin() + 10, prompt.begin() + 25);
    lookup.append(repeated);
    check(lookup.propose(25) == std::vector<uint32_t>({26, 27, 28, 29, 30, 31, 32}),
          "the pending anchor must finish the lookup suffix");
    check(lookup.propose(25, 3) == std::vector<uint32_t>({26, 27, 28}), "proposal cap");
    check(lookup.propose(26).empty(), "token equality must verify the entire suffix");
    lookup.reset({});
    check(lookup.propose(25).empty(), "reset must discard prior request tokens");
    std::vector<uint32_t> common(32);
    std::iota(common.begin(), common.end(), 0);
    std::vector<uint32_t> ambiguousPrompt{100};
    ambiguousPrompt.insert(ambiguousPrompt.end(), common.begin(), common.end());
    ambiguousPrompt.push_back(200);
    ambiguousPrompt.push_back(101);
    ambiguousPrompt.insert(ambiguousPrompt.end(), common.begin(), common.end());
    ambiguousPrompt.push_back(201);
    lookup.reset(ambiguousPrompt);
    lookup.append(std::array<uint32_t, 1>{100});
    lookup.append(std::span(common).first(31));
    check(lookup.propose(31, 1) == std::vector<uint32_t>{200},
          "longest suffix must disambiguate identical function bodies");
    PromptLookup bounded(4, 8, 80);
    bounded.reset(prompt);
    check(bounded.indexedSuffixes() == 8, "index must stay bounded");
    bounded.append(repeated);
    bounded.append(std::array<uint32_t, 2>{5, 6});
    check(bounded.historyTokens() == 0 && bounded.propose(25).empty(), "history overflow falls back");
    VerifyRowController controller;
    for (uint32_t width : VerifyRowController::widths) {
      for (uint32_t trial = 0; trial < 2; ++trial) {
        check(controller.choose(1024) == width, "all widths need independent complete-round samples");
        controller.observe(width, width - 1, width == 1 ? 100.0 : 10.0, 1024);
      }
    }
    check(controller.choose(1024) == 8, "flat costs and full acceptance prefer widest block");
    check(controller.choose(8192) == 8, "new context bucket needs fresh calibration");
    VerifyRowController strongAcceptance(true);
    for (uint32_t round = 0; round < 48; ++round) {
      check(strongAcceptance.choose(1024) == 8,
            "strong M5 M8 acceptance must not spend short answers probing narrower widths");
      strongAcceptance.observe(8, round % 2 ? 7 : 6, 10.0, 1024);
    }
    VerifyRowController decliningAcceptance(true);
    for (uint32_t round = 0; round < 8; ++round)
      decliningAcceptance.observe(8, 7, 10.0, 1024);
    decliningAcceptance.observe(8, 0, 10.0, 1024);
    check(decliningAcceptance.choose(1024) == 8,
          "one poor round must not discard a strong recent acceptance window");
    decliningAcceptance.observe(8, 0, 10.0, 1024);
    check(decliningAcceptance.choose(1024) == 4,
          "falling wide acceptance must unlock four-row measurement first");
    VerifyRowController lowAcceptance(true);
    const std::array<double, 4> actualCosts{10, 8, 3, 2};
    for (size_t index = 0; index < VerifyRowController::widths.size(); ++index) {
      const uint32_t width = VerifyRowController::widths[index];
      for (uint32_t trial = 0; trial < 2; ++trial) {
        check(lowAcceptance.choose(1024) == width,
              "low acceptance needs actual cost probes in descending width order");
        lowAcceptance.observe(width, 0, actualCosts[index], 1024);
      }
    }
    check(lowAcceptance.choose(1024) == 1,
          "low acceptance must adapt to the fastest measured complete round");
    VerifyRowController flatWideBest(true);
    for (size_t index = 0; index < VerifyRowController::widths.size(); ++index)
      for (uint32_t trial = 0; trial < 2; ++trial)
        flatWideBest.observe(VerifyRowController::widths[index], 0,
                             double(uint32_t{1} << index), 1024);
    check(flatWideBest.choose(1024) == 8,
          "poor acceptance alone must not force a slower measured narrow graph");
    check(strongAcceptance.choose(8192) == 8,
          "a new context bucket starts with the wide graph");
    strongAcceptance.observe(8, 0, 10.0, 8192);
    check(strongAcceptance.choose(8192) == 8,
          "a new context bucket needs two actual wide rounds");
    strongAcceptance.observe(8, 0, 10.0, 8192);
    check(strongAcceptance.choose(8192) == 4,
          "old strong acceptance must not suppress calibration in a new bucket");
    bool rejected = false;
    try { controller.observe(4, 4, 1, 1024); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "acceptance cannot exceed proposal count");
    rejected = false;
    try { controller.observe(4, 0, std::numeric_limits<double>::quiet_NaN(), 1024); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "nonfinite timing must be rejected");
    std::puts("decode policy tests passed");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
