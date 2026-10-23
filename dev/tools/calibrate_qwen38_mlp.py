#!/usr/bin/env python3
"""Score Qwen3.8 SwiGLU channels on a small local coding/agent calibration set.

Requires MLX and mlx-lm on Apple Silicon. It opens one local checkpoint,
records LayerTrim-style expected channel contribution scores, and emits JSON;
it does not edit or re-quantize model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen3_next


def read_records(path: Path) -> list[dict]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"no calibration records in {path}")
    for record in records:
        if not isinstance(record.get("messages"), list) or not record["messages"]:
            raise ValueError(f"record {record.get('id')!r} has no messages")
    return records


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
        "--max-tokens", type=int, default=4096, help="refuse longer individual records"
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    records = read_records(args.calibration.expanduser().resolve())
    print(f"Loading one local checkpoint: {model_path}", flush=True)
    model, tokenizer = load(str(model_path))
    text_model = model.model
    layers = text_model.layers
    if len(layers) != 64:
        raise ValueError(f"expected 64 Qwen3.8 layers, found {len(layers)}")
    width = int(model.args.text_config["intermediate_size"])
    if width != 17408 or width % 64:
        raise ValueError(
            f"expected source intermediate width 17408 aligned to 64, found {width}"
        )
    if any(not hasattr(layer.mlp, "gate_proj") for layer in layers):
        raise ValueError("expected a dense SwiGLU MLP in every target layer")

    print(
        f"Loaded {len(layers)} layers, width {width}; "
        f"MLX active {mx.get_active_memory() / 2**30:.2f} GiB",
        flush=True,
    )

    # LayerTrim's feature score uses E[phi_j^2] * ||W_down[:, j]||^2.
    # Dequantize one 5120 x 17408 down matrix at a time, immediately reduce it
    # to 17408 column norms, and retain only those small vectors on host RAM.
    down_column_norm_sq = []
    for index, layer in enumerate(layers):
        projection = layer.mlp.down_proj
        dense = mx.dequantize(
            projection.weight,
            scales=projection.scales,
            biases=projection.biases,
            group_size=projection.group_size,
            bits=projection.bits,
            mode=projection.mode,
        )
        norm_sq = mx.sum(dense.astype(mx.float32) ** 2, axis=0)
        mx.eval(norm_sq)
        values = np.asarray(norm_sq, dtype=np.float64)
        if values.shape != (width,):
            raise ValueError(
                f"layer {index} down-projection norm shape is {values.shape}"
            )
        down_column_norm_sq.append(values)
        del dense, norm_sq, values
        if index % 16 == 15:
            print(f"Prepared down-projection norms through layer {index}", flush=True)

    activation_tokens = np.zeros(len(layers), dtype=np.int64)
    category_energy: dict[str, np.ndarray] = {}
    category_tokens: dict[str, np.ndarray] = {}
    current_category = "uncategorized"
    block_influence_sum = np.zeros(len(layers), dtype=np.float64)
    block_influence_tokens = np.zeros(len(layers), dtype=np.int64)
    mlp_ids = {id(layer.mlp): index for index, layer in enumerate(layers)}
    layer_ids = {id(layer): index for index, layer in enumerate(layers)}
    mlp_type = type(layers[0].mlp)
    decoder_type = type(layers[0])
    original_mlp_call = mlp_type.__call__
    original_decoder_call = decoder_type.__call__

    def score_mlp(self, x):
        index = mlp_ids.get(id(self))
        if index is None:
            return original_mlp_call(self, x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        phi = qwen3_next.swiglu(gate, up)
        squared = mx.sum(phi.astype(mx.float32) ** 2, axis=(0, 1))
        mx.eval(squared)
        activation_tokens[index] += x.shape[0] * x.shape[1]
        category_energy[current_category][index] += np.asarray(
            squared, dtype=np.float64
        )
        category_tokens[current_category][index] += x.shape[0] * x.shape[1]
        return self.down_proj(phi)

    def score_decoder(self, x, mask=None, cache=None):
        index = layer_ids.get(id(self))
        out = original_decoder_call(self, x, mask=mask, cache=cache)
        if index is not None:
            left, right = x.astype(mx.float32), out.astype(mx.float32)
            cosine = mx.sum(left * right, axis=-1) / (
                mx.sqrt(mx.sum(left * left, axis=-1) + 1e-12)
                * mx.sqrt(mx.sum(right * right, axis=-1) + 1e-12)
            )
            mx.eval(cosine)
            block_influence_sum[index] += float(np.asarray(mx.sum(1.0 - cosine)))
            block_influence_tokens[index] += x.shape[0] * x.shape[1]
        return out

    model.eval()
    mlp_type.__call__ = score_mlp
    decoder_type.__call__ = score_decoder
    ids_used: list[str] = []
    total_tokens = 0
    try:
        for record in records:
            current_category = str(record.get("category", "uncategorized"))
            if current_category not in category_energy:
                category_energy[current_category] = np.zeros(
                    (len(layers), width), dtype=np.float64
                )
                category_tokens[current_category] = np.zeros(
                    len(layers), dtype=np.int64
                )
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
            total_tokens += len(token_ids)
            ids_used.append(str(record.get("id", len(ids_used))))
            print(
                f"Scored {ids_used[-1]}: {len(token_ids)} tokens; "
                f"MLX active {mx.get_active_memory() / 2**30:.2f} GiB",
                flush=True,
            )
    finally:
        mlp_type.__call__ = original_mlp_call
        decoder_type.__call__ = original_decoder_call

    if np.any(activation_tokens == 0):
        missing = np.flatnonzero(activation_tokens == 0).tolist()
        raise RuntimeError(f"no MLP activations captured for layers {missing}")

    down_norms = np.stack(down_column_norm_sq)
    category_channel_scores = {
        category: values / category_tokens[category][:, None] * down_norms
        for category, values in category_energy.items()
    }
    if len(category_channel_scores) < 2:
        raise ValueError("calibration set must contain at least two task categories")
    # Balance task families explicitly instead of allowing the longer corpus
    # category to dominate every layer's channel ranking.
    normalized_category_scores = {
        category: values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)
        for category, values in category_channel_scores.items()
    }
    balanced_channel_scores = np.mean(
        np.stack(list(normalized_category_scores.values())), axis=0
    )
    group_scores = balanced_channel_scores.reshape(len(layers), width // 64, 64).sum(
        axis=2
    )
    category_group_scores = {
        category: values.reshape(len(layers), width // 64, 64).sum(axis=2)
        for category, values in category_channel_scores.items()
    }
    block_influence = 1.0 - block_influence_sum / np.maximum(block_influence_tokens, 1)
    result = {
        "method": "LayerTrim-style expected squared SwiGLU contribution; activation-weighted down-projection column norm",
        "source_model": str(model_path),
        "calibration_file": str(args.calibration.expanduser().resolve()),
        "calibration_ids": ids_used,
        "calibration_tokens": total_tokens,
        "calibration_tokens_by_category": {
            category: int(tokens[0]) for category, tokens in category_tokens.items()
        },
        "original_layers": len(layers),
        "intermediate_size": width,
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        "group_count_per_layer": width // 64,
        "channel_score_definition": "per-category mean_token(silu(gate(x))*up(x))^2 * sum_hidden(dequantized_down_weight^2); normalize each category's channel scores to unit sum per layer, then average categories equally for selection",
        "block_influence_definition": "mean_token(1 - cosine(decoder_layer_input, decoder_layer_output))",
        "layers": [
            {
                "layer": index,
                "mlp_tokens": int(activation_tokens[index]),
                "block_influence": float(block_influence[index]),
                "group_scores": [float(score) for score in group_scores[index]],
                "category_group_scores": {
                    category: [float(score) for score in values[index]]
                    for category, values in category_group_scores.items()
                },
            }
            for index in range(len(layers))
        ],
    }
    args.output.expanduser().resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"Wrote calibrated scores for {total_tokens} tokens across {len(layers)} layers "
        f"to {args.output.expanduser().resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
