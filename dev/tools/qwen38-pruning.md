# Qwen3.8-27B pruning candidates

`prune_qwen38.py` makes local, text-only checkpoints from a cached MLX
Qwen3.8-27B checkpoint. It keeps whole four-layer hybrid groups, so every
candidate still has the runtime's three GDN layers followed by one full
attention layer. It preserves the five groups containing the published
DFlash2 taps, compacts the layer numbers, and writes those new tap numbers to
both the target config and a candidate-local copy of the DFlash config.

The cached base has 26.896B logical text parameters. The prepared candidates
are:

| Candidate | Layers | Logical text parameters | Reduction | Weight files | Remapped taps |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen38-28l` | 28 | 13.197B | 50.9% | 6.92 GiB | `[5, 15, 17, 23, 25]` |
| `qwen38-24l` | 24 | 11.675B | 56.6% | 6.12 GiB | `[5, 11, 13, 19, 21]` |
| `qwen38-20l` | 20 | 10.153B | 62.2% | 5.32 GiB | `[1, 7, 9, 15, 17]` |

The counts are based on logical tensor shapes, excluding vision and quantization
scales. Each candidate also records the exact retained original groups and
layer IDs in its `pruning_manifest.json`. The 20-layer candidate is the
smallest one that
retains all five tapped groups under this group-pruning constraint. Going
lower requires changing which activations DFlash reads, pruning within groups,
or training a new draft; this tool does none of those.

The extraction copies already-quantized tensors without dequantizing or
requantizing them. It does no layer scoring, calibration, recovery training or
distillation. Remapping tap indices makes the files geometrically loadable;
it does not make the original DFlash selector accurate on the changed target.
Treat all candidates as experiments, and compare coding, tool-use and
termination quality against the unpruned 4-bit Splash checkpoint before
choosing one.

## Build

Use local snapshots already in the Hugging Face cache:

```sh
.venv/bin/python dev/tools/prune_qwen38.py \
  --source ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/<commit> \
  --draft ~/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/<commit> \
  --output ~/Models/Splash/Qwen3.8-pruned
```

The tool uses only Python's standard library. The source checkpoint remains
untouched. Candidate weights are written outside
the checkout; draft weight files are symlinked to the cached source.

## Serve a candidate

Build Splash, then use the candidate's local directory and local draft path:

```sh
./splash serve \
  --model ~/Models/Splash/Qwen3.8-pruned/qwen38-24l \
  --draft-model ~/Models/Splash/Qwen3.8-pruned/qwen38-24l/draft \
  --language-only
```

Repeat with `qwen38-28l` or `qwen38-20l`. `--language-only` is required
because the candidates omit the vision tower. The full model and its published
DFlash2 draft remain the baseline.
