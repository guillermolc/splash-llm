#!/usr/bin/env python3
"""Choose Qwen3.8 MLP groups by calibrated block-output reconstruction.

For each 64-channel quantization group, this measures the group's actual
contribution to the SwiGLU down-projection output. A greedy set selection then
removes the groups that add the least error to a category-balanced
reconstruction objective. The resulting JSON is accepted by
prune_qwen38_mlp.py. This is a local, training-free calibration procedure; it
does not update or requantize checkpoint weights.

Requires MLX and mlx-lm on Apple Silicon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen3_next

WIDTH = 17408
GROUP_SIZE = 64
GROUP_COUNT = WIDTH // GROUP_SIZE
ORIGINAL_LAYERS = 64


def read_records(path: Path) -> list[dict]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"no calibration records in {path}")
    for record in records:
        if not isinstance(record.get("messages"), list) or not record["messages"]:
            raise ValueError(f"record {record.get('id')!r} has no messages")
        if not record.get("category"):
            raise ValueError(f"record {record.get('id')!r} has no category")
    return records


def greedy_keep_mask(
    grams: dict[str, np.ndarray], keep_count: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Greedily remove groups minimizing equal-weight relative output error."""
    if not 0 < keep_count < GROUP_COUNT:
        raise ValueError(f"keep_count must be in 1..{GROUP_COUNT - 1}")
    balanced = np.zeros((GROUP_COUNT, GROUP_COUNT), dtype=np.float64)
    normalized: dict[str, np.ndarray] = {}
    ones = np.ones(GROUP_COUNT, dtype=np.float64)
    for category, gram in grams.items():
        full_energy = float(ones @ gram @ ones)
        if not np.isfinite(full_energy) or full_energy <= 1e-30:
            raise ValueError(
                f"invalid full MLP output energy for {category}: {full_energy}"
            )
        normalized[category] = gram / full_energy
        balanced += normalized[category] / len(grams)

    remove_count = GROUP_COUNT - keep_count
    removed: list[int] = []
    available = np.ones(GROUP_COUNT, dtype=bool)
    marginal = np.diag(balanced).copy()
    for _ in range(remove_count):
        candidates = np.where(available, marginal, np.inf)
        selected = int(np.argmin(candidates))
        available[selected] = False
        removed.append(selected)
        marginal += 2.0 * balanced[:, selected]

    removed_vector = np.zeros(GROUP_COUNT, dtype=np.float64)
    removed_vector[removed] = 1.0
    relative_errors = {
        category: float(removed_vector @ gram @ removed_vector)
        for category, gram in normalized.items()
    }
    return available, relative_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=Path, help="local MLX Qwen3.8 checkpoint"
    )
    parser.add_argument(
        "--calibration", required=True, type=Path, help="coding/agent JSONL records"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="score JSON destination"
    )
    parser.add_argument(
        "--keep-width",
        type=int,
        default=14336,
        help="candidate MLP width, multiple of 64",
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=8, help="tokens per reconstruction chunk"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="refuse longer individual records"
    )
    args = parser.parse_args()

    if args.keep_width <= 0 or args.keep_width >= WIDTH or args.keep_width % GROUP_SIZE:
        raise ValueError(f"--keep-width must be a multiple of 64 below {WIDTH}")
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    model_path = args.model.expanduser().resolve()
    records = read_records(args.calibration.expanduser().resolve())
    manifest_path = model_path / "pruning_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"expected pruning manifest at {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text())
    original_layer_ids = [
        int(value) for value in source_manifest["retained_original_layers"]
    ]

    print(f"Loading one local checkpoint: {model_path}", flush=True)
    model, tokenizer = load(str(model_path))
    layers = model.model.layers
    if len(layers) != len(original_layer_ids):
        raise ValueError(
            f"manifest has {len(original_layer_ids)} layers, MLX loaded {len(layers)}"
        )
    width = int(model.args.text_config["intermediate_size"])
    if width != WIDTH:
        raise ValueError(f"expected source width {WIDTH}, found {width}")

    categories = sorted({str(record["category"]) for record in records})
    if len(categories) < 2:
        raise ValueError("calibration set must contain at least two categories")
    grams_by_category = {
        category: [
            np.zeros((GROUP_COUNT, GROUP_COUNT), dtype=np.float64) for _ in layers
        ]
        for category in categories
    }
    tokens_by_category = {category: 0 for category in categories}
    layer_tokens = np.zeros(len(layers), dtype=np.int64)
    mlp_ids = {id(layer.mlp): index for index, layer in enumerate(layers)}
    mlp_type = type(layers[0].mlp)
    original_mlp_call = mlp_type.__call__
    current_category = categories[0]

    def score_mlp(self, x):
        index = mlp_ids.get(id(self))
        if index is None:
            return original_mlp_call(self, x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        phi = qwen3_next.swiglu(gate, up).reshape((-1, WIDTH)).astype(mx.float32)
        projection = self.down_proj
        dense_down = mx.dequantize(
            projection.weight,
            scales=projection.scales,
            biases=projection.biases,
            group_size=projection.group_size,
            bits=projection.bits,
            mode=projection.mode,
        ).astype(mx.float32)
        # MLX affine weights are [hidden, intermediate]. Group contributions
        # are [token, group, hidden]; their sum is the layer's FFN output.
        down_groups = mx.transpose(dense_down, (1, 0)).reshape(
            (GROUP_COUNT, GROUP_SIZE, 5120)
        )
        gram = np.zeros((GROUP_COUNT, GROUP_COUNT), dtype=np.float64)
        token_count = int(phi.shape[0])
        for start in range(0, token_count, args.chunk_tokens):
            end = min(start + args.chunk_tokens, token_count)
            grouped_phi = phi[start:end].reshape((-1, GROUP_COUNT, GROUP_SIZE))
            contributions = mx.einsum("tgc,gch->tgh", grouped_phi, down_groups)
            chunk_gram = mx.einsum("tgh,tkh->gk", contributions, contributions)
            mx.eval(chunk_gram)
            gram += np.asarray(chunk_gram, dtype=np.float64)
            del grouped_phi, contributions, chunk_gram
        grams_by_category[current_category][index] += gram
        layer_tokens[index] += token_count
        del gram, phi, dense_down, down_groups
        return original_mlp_call(self, x)

    model.eval()
    mlp_type.__call__ = score_mlp
    calibration_ids: list[str] = []
    total_tokens = 0
    try:
        for record in records:
            current_category = str(record["category"])
            kwargs = {
                "tokenize": True,
                "add_generation_prompt": False,
                "reasoning_effort": "medium",
            }
            if record.get("tools"):
                kwargs["tools"] = record["tools"]
            tokens = tokenizer.apply_chat_template(record["messages"], **kwargs)
            token_ids = [int(token) for token in tokens]
            if not token_ids:
                raise ValueError(f"record {record.get('id')!r} rendered to no tokens")
            if len(token_ids) > args.max_tokens:
                raise ValueError(
                    f"record {record.get('id')!r} has {len(token_ids)} tokens, "
                    f"above --max-tokens={args.max_tokens}"
                )
            logits = model(mx.array([token_ids]))
            mx.eval(logits)
            del logits
            count = len(token_ids)
            total_tokens += count
            tokens_by_category[current_category] += count
            calibration_ids.append(str(record.get("id", len(calibration_ids))))
            print(
                f"Reconstructed {calibration_ids[-1]}: {count} tokens; "
                f"MLX active {mx.get_active_memory() / 2**30:.2f} GiB",
                flush=True,
            )
    finally:
        mlp_type.__call__ = original_mlp_call

    if np.any(layer_tokens == 0):
        missing = np.flatnonzero(layer_tokens == 0).tolist()
        raise RuntimeError(f"no MLP activations reconstructed for layers {missing}")

    layer_results: dict[int, dict] = {}
    for index, original_layer in enumerate(original_layer_ids):
        layer_grams = {
            category: grams_by_category[category][index] for category in categories
        }
        keep_mask, relative_errors = greedy_keep_mask(
            layer_grams, args.keep_width // GROUP_SIZE
        )
        scores = np.zeros(GROUP_COUNT, dtype=np.float64)
        scores[keep_mask] = 1.0
        layer_results[original_layer] = {
            "layer": original_layer,
            "mlp_tokens": int(layer_tokens[index]),
            "group_scores": scores.tolist(),
            "retained_group_ids": np.flatnonzero(keep_mask).tolist(),
            "relative_block_output_mse": relative_errors,
        }

    output_layers = []
    for original_layer in range(ORIGINAL_LAYERS):
        if original_layer in layer_results:
            output_layers.append(layer_results[original_layer])
        else:
            output_layers.append(
                {
                    "layer": original_layer,
                    "mlp_tokens": 0,
                    "group_scores": [0.0] * GROUP_COUNT,
                    "retained_group_ids": [],
                    "relative_block_output_mse": {},
                }
            )

    result = {
        "method": "category-balanced greedy group selection minimizing calibrated SwiGLU block-output reconstruction error",
        "source_model": str(model_path),
        "calibration_file": str(args.calibration.expanduser().resolve()),
        "calibration_ids": calibration_ids,
        "calibration_tokens": total_tokens,
        "calibration_tokens_by_category": tokens_by_category,
        "original_layers": ORIGINAL_LAYERS,
        "source_retained_original_layers": original_layer_ids,
        "intermediate_size": WIDTH,
        "selection_width": args.keep_width,
        "quantization": {"mode": "affine", "bits": 4, "group_size": GROUP_SIZE},
        "group_count_per_layer": GROUP_COUNT,
        "group_scores_definition": "1 for greedily retained groups and 0 for removed groups; scores encode a fixed reconstruction-optimal mask for prune_qwen38_mlp.py",
        "layers": output_layers,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote reconstruction scores to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
