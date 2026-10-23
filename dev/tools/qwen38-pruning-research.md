# Qwen3.8 pruning and DFlash2: current evidence

Updated 2026-10-23. This note records the local pruning frontier, Splash
kernel constraints, and the work needed to make a credible quality claim.

## What has actually been tested

All pruning candidates tested so far were derived from
`mlx-community/Qwen3.8-27B-4bit`. They use exact Q4 group slicing, with no
requantization, target fine-tuning, or distillation. No BF16 Qwen3.8-27B
checkpoint is present in the local Hugging Face cache. The candidates therefore
say nothing conclusive about pruning the BF16 teacher and then quantizing it.

Splash's MLX target loader expects quantized linear projections; an unquantized
BF16 Transformers checkpoint is not a directly serveable target. A BF16 source
can still be useful for calibration and distillation, followed by export to a
Splash-supported quantized checkpoint.

| Candidate | Logical parameters | Fixed coding / review / agent gates | Finding |
| --- | ---: | --- | --- |
| 52 layers, MLP 14,336 | 19.876B | pass / pass / pass | Smallest tested candidate passing all three local gates |
| 52 layers, MLP 13,312 | 19.058B | fail / pass / pass | Coding answer stopped in a long explanation instead of returning code |
| 64 layers, MLP 10,240 | 19.850B | fail / pass / pass | Keeping all layers did not rescue the coding gate |
| 52 layers, MLP 14,080 on non-tap layers, tap MLPs full | 19.927B | pass / pass / pass | DFlash acceptance fell to 0.619; ANE split was unavailable |
| 48- and 44-layer combinations | 20.808B and 19.286B | no combination passed all three | More layer removal currently hurts at least one gate |

The gates are intentionally narrow: one constrained Python implementation,
one constrained code-review function, and a three-step tool workflow whose tool
results are simulated. They are useful screening checks, not evidence that a
candidate retains the base model's benchmark quality.

## Decode and draft results

The same-build M5 Max comparison used the native backend benchmark, a 39-token
prompt, 64 generated tokens per lane, three samples per round, and ABBA order.
The 52-layer full-MLP reference had DFlash acceptance 0.873; the 14,336-width
candidate had 0.771. The candidate's measured median throughput was lower by
about 5-6% at batch widths 1-2. Widths 3-4 were inconclusive under Splash's
noise rule. The separate comparison with the 64-layer base had 26-28% round
spread and is discarded. These short-prompt runs do not establish performance
for normal coding contexts.

This points to a specific bottleneck in the current experiment: narrower target
MLPs reduce target compute, while the unchanged base-trained draft accepts
fewer tokens. The reduction in target work did not yet compensate for the
acceptance loss. A ridge fit to DFlash's context projection (`fc.weight`)
improved its training-set feature reconstruction but left held-out error high
and did not change measured acceptance. More fitting of `fc.weight` alone is
not the next useful experiment.

## Splash and kernel constraints

- Splash's GPU linear plans use the actual projection dimensions. The tested
  widths are compatible with the GPU path when output widths are multiples of
  256 and input widths match the quantization group geometry.
- `AneFfn::unsupported()` requires intermediate widths divisible by 512,
  at least 1,024 channels, and identical gate/up/down shapes across all layers.
  The 14,080-width mixed profile violates both the 512-channel and uniform
  shape requirements, so the prefill FFN split stays on GPU. Heterogeneous ANE
  support would need per-layer program shapes, weight staging and scratch
  planning. The current decode bottleneck is the GPU target FFN, so this is
  secondary to recovering draft acceptance.
- The current GPU profiler groups dispatches by pipeline name. Target and
  draft share linear/MLP pipelines, so it cannot attribute their total costs
  separately. Its percentages also use isolated dispatch replay, not fused
  cycle time. The earlier claim that the draft costs less than 1% conflated
  draft-specific kernels with the whole draft and is withdrawn. Target MLP
  work is structurally reduced by narrowing, but identifying the dominant
  measured owner requires operation/owner attribution before kernel tuning.
- Keeping the five DFlash tap layers' MLPs full width did not preserve the
  draft's feature distribution. Every tap activation depends on all preceding
  residual and MLP computations, so tap-layer protection alone is insufficient.

## Draft adaptation path

Reviewing Splash's open PRs exposed a more direct path than the custom MLX
trainer proposed in the previous version of this note. These PRs were checked
on 2026-10-23; none of the open training PRs is merged into this branch.

| PR | Contribution | Limit for this experiment |
| --- | --- | --- |
| [499](https://github.com/incoai/splash/pull/499) | DFlash2 selector/full training, cached teacher features, BF16 export | Validates released family geometries and loads a BF16 Transformers teacher |
| [537](https://github.com/incoai/splash/pull/537) | Apple Silicon MPS training | Small component checks; full 27B MPS capacity/quality is unqualified |
| [551](https://github.com/incoai/splash/pull/551) | Loader-matched AF4G64 projection QAT | Draft weight quantization, not a Q4 teacher adapter |
| [552](https://github.com/incoai/splash/pull/552) | Prefix survival and position-weighted losses | Training objective; needs representative native validation |
| [553](https://github.com/incoai/splash/pull/553) | Pair-context and nonlinear selectors | Research exports explicitly mark `native_compatible: false` |
| [554](https://github.com/incoai/splash/pull/554) | Context/late-backbone/recall training scopes | Four short published runs retained validation 3.875 to 3.875; no demonstrated gain |
| [550](https://github.com/incoai/splash/pull/550) and [556](https://github.com/incoai/splash/pull/556) | Native candidates, scores and final draft hidden capture | Enables selector adaptation against our actual deployed Q4 target |

The training chain is not directly usable on this checkpoint. The inspected
499 head (`a4380587`) validates the released target family; the 554 head
(`310e75d7`) explicitly expects 64 target layers and MLP 17,408. Neither
accepts our 52-layer Q4 target as a teacher. Its command-line path also loads
the teacher even when feature caches already exist. MPS support nevertheless
corrects the earlier statement that Apple training necessarily requires
writing a new trainer from scratch.

The immediate local path uses the ideas in 550 (`ef0e95d4`) and 556
(`b4006b12`): capture the deployed draft's normalized BF16 hidden rows,
natural top-16 candidates, unary scores, transition edges and verifier
outcomes. The existing arenas already contain these tensors. No shader or
production decode change is needed. Only verified positions through the
first rejection are supervised; later verifier rows are censored because they
were conditioned on a wrong proposal. Gold tokens must never be inserted into
the natural candidate set when reporting recall or evaluating proposals.

This separates candidate recall failures from selector errors. If recall is
adequate, first train only `candidate_selector.hidden_projection.weight`:
5120 x 256 = 1,310,720 parameters. The two vocabulary codebooks stay frozen;
training all selector tensors would involve about 128.45M parameters. Use
loader-matched projection QAT, BF16 projection output and FP32 score
accumulation. Capture replay is a diagnostic and training source; every
candidate must subsequently be rerun in the native engine on held-out prompts
because changed proposals change future draft context and state.

If gold is absent from top-16, reranking cannot repair that rejection.
Adapting context/backbone then needs target-tap feature sequences and token/KL
losses, following the selective recall recipe. Frozen final hidden captures
alone cannot train the earlier draft layers.

NVIDIA NeMo AutoModel now has a DFlash2 training recipe for Qwen3.8. Its
objective supervises block positions 1 through 7 with position-decayed token
cross-entropy and adds the DFlash2 candidate-selector cross-entropy. The Qwen3.8
recipe uses block size 8, 64 anchors, selector top-k 16 and loss decay gamma 4.
It also requires training responses regenerated by the target to reduce the
teacher-forcing/inference distribution mismatch.

For a 52-layer target, training must use the remapped taps `[5, 15, 25, 39, 49]`
and the pruned target's own regenerated coding/agentic conversations. The
published NeMo launch path uses PyTorch distributed/NCCL and a two-GPU target
configuration; it is not a drop-in Apple Silicon trainer. Speculators also
provides experimental DFlash2 training with strict candidate-miss masking,
but its documented extraction/training path uses CUDA/vLLM. MLX is installed
in the isolated pruning environment and supports autodiff. A small cached
selector fit can use MLX without loading the target. The earlier
context-projection-only ridge fit is not a substitute for token/selector
training.

## Quality target and next sequence

The subsequent native-capture pilot is documented in
[dflash-adaptation.md](dflash-adaptation.md). It implemented a small MLX
projection-only QAT path, without loading a BF16 teacher. Of 195 verified
rejections from the pruned target, 151 had gold present in its natural top-16.
The first 100-step fit selected step 80 on four validation prompts. Native
ABBA reruns produced identical 515-token outputs using 155 rather than 159
cycles per round. This is a small diagnostic gain, not recovered target
quality or evidence of a general serving speedup. The separate experimental
draft is available for further native evaluation. The target remains the
19.876B screening winner; no additional pruning has passed a stronger
quality evaluation yet.

The project contains runtime performance benchmarks, but no SWE-bench,
Terminal-Bench, or LiveCodeBench quality harness. The official Qwen model card
reports 73.0 on Terminal-Bench 2.1, 61.7 on SWE-bench Pro, 42.3 on
NL2Repo-Bench, 42.2 on DeepSWE 1.1, 79.0 on QwenSWEBench and 90.3 on
LiveCodeBench v6. QwenSWEBench is in-house; the card describes the harness and
scoring settings for the other code/agent rows. Before claiming quality parity,
evaluate the same quantized 64-layer Splash baseline and each pruned candidate
with a reproducible coding/agentic suite; compare multiple seeds and report
uncertainty.

The next technically sound sequence is:

1. Keep the 19.876B candidate as the current Q4 screening winner, not as a
   quality-equivalent replacement.
2. Measure native candidate recall and selector errors on the current Q4
   candidate, and attempt a cached selector projection fit only if its
   measured headroom justifies it.
3. Use a BF16 teacher for activation/error analysis and student distillation;
   prune the target's MLPs with layer-specific output reconstruction and
   coding/agentic calibration, then quantize the student to Splash's supported
   MLX affine format.
4. Train the complete DFlash2 draft against the resulting pruned target using
   on-policy responses, the remapped taps and both DFlash2 losses.
5. Run the quality suite before the next pruning step. Then use same-build,
   idle ABBA runs with realistic prompt lengths to measure target GPU cycles,
   acceptance, and end-to-end throughput separately.
6. Tune kernels only where the new target's GPU profile still shows material
   time. Retain uniform widths divisible by 512 when the prefill ANE split is
   part of the desired workload.

## References

- [Qwen3.8-27B official model card and benchmark methodology](https://huggingface.co/Qwen/Qwen3.8-27B)
- [NVIDIA NeMo DFlash and DFlash2 training guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/speculative/dflash.mdx)
- [Qwen3.8-27B DFlash2 training recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/speculative/dflash/qwen3_8_27b_dflash2.yaml)
- [z-lab DFlash MLX inference implementation](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py)
- [Speculators DFlash2 training and strict candidate selection](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md)
- [Published short native qualification of the Splash training PRs](https://github.com/linson007/splash/blob/310e75d7f22e7a86f2141c5cf93e82aba62a9342/docs/experiments/qwen38-a100-smoke/README.md)
