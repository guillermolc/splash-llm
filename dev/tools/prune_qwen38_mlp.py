#!/usr/bin/env python3
"""Build calibrated Qwen3.8 checkpoints with narrower SwiGLU MLPs.

Selects complete 64-channel MLX affine-Q4 input groups using scores from
calibrate_qwen38_mlp.py. It preserves the source layer set and DFlash taps,
whether the source has 64 layers or is an existing layer-pruned candidate. It
slices packed projection tensors without requantization and does not fine-tune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
from prune_qwen38 import (
    TOKENIZER_FILES,
    logical_parameter_count,
    read_json,
    read_safetensors_header,
    source_shards,
    write_json,
)

BASE_WIDTH = 17408
CHANNELS_PER_GROUP = 64
PACKED_VALUES_PER_WORD = 8
BASE_TAPS = (5, 19, 33, 47, 61)
ORIGINAL_LAYERS = 64
DTYPES = {
    "BOOL": np.dtype("u1"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("V2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "F32": np.dtype("<f4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F64": np.dtype("<f8"),
}


def layer_projection(name: str) -> tuple[int, str, str] | None:
    parts = name.split(".")
    if len(parts) < 7 or parts[:3] != ["language_model", "model", "layers"]:
        return None
    try:
        layer = int(parts[3])
    except ValueError:
        return None
    if parts[4] != "mlp" or parts[5] not in {"gate_proj", "up_proj", "down_proj"}:
        return None
    return layer, parts[5], ".".join(parts[6:])


def selected_shape(name: str, shape: list[int], layer_widths: list[int]) -> list[int]:
    projection = layer_projection(name)
    if projection is None or projection[2] not in {"weight", "scales", "biases"}:
        return list(shape)
    layer, module, tensor_kind = projection
    width = layer_widths[layer]
    if len(shape) != 2:
        raise ValueError(f"unexpected {module} tensor rank for {name}: {shape}")
    if module in {"gate_proj", "up_proj"}:
        if shape[0] != BASE_WIDTH:
            raise ValueError(f"unexpected gate/up shape for {name}: {shape}")
        return [width, shape[1]]
    if tensor_kind == "weight":
        if tuple(shape) != (5120, BASE_WIDTH // PACKED_VALUES_PER_WORD):
            raise ValueError(f"unexpected down packed weight shape for {name}: {shape}")
        return [shape[0], width // PACKED_VALUES_PER_WORD]
    if tuple(shape) != (5120, BASE_WIDTH // CHANNELS_PER_GROUP):
        raise ValueError(f"unexpected down quantization shape for {name}: {shape}")
    return [shape[0], width // CHANNELS_PER_GROUP]


def selected_tensor(
    name: str,
    entry: dict,
    mapped: mmap.mmap,
    data_start: int,
    row_ids: np.ndarray,
    word_ids: np.ndarray,
    group_ids: np.ndarray,
    layer_widths: list[int],
) -> tuple[list[int], bytes] | None:
    projection = layer_projection(name)
    if projection is None:
        return None
    _, module, tensor_kind = projection
    if tensor_kind not in {"weight", "scales", "biases"}:
        return None
    shape = entry["shape"]
    selected_shape(name, shape, layer_widths)
    if len(shape) != 2:
        raise ValueError(f"unexpected {module} tensor rank for {name}: {shape}")
    dtype = DTYPES.get(entry["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype {entry['dtype']} for {name}")
    start, end = entry["data_offsets"]
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if end - start != expected_bytes:
        raise ValueError(f"tensor byte count does not match shape for {name}")
    tensor = np.frombuffer(
        mapped,
        dtype=dtype,
        count=int(np.prod(shape, dtype=np.int64)),
        offset=data_start + start,
    ).reshape(shape)

    if module in {"gate_proj", "up_proj"}:
        if tensor_kind == "weight" and tuple(shape) != (BASE_WIDTH, 640):
            raise ValueError(
                f"unexpected gate/up packed weight shape for {name}: {shape}"
            )
        selected = np.take(tensor, row_ids, axis=0)
        new_shape = list(selected.shape)
    else:
        if tensor_kind == "weight":
            if tuple(shape) != (5120, BASE_WIDTH // PACKED_VALUES_PER_WORD):
                raise ValueError(
                    f"unexpected down packed weight shape for {name}: {shape}"
                )
            selected = np.take(tensor, word_ids, axis=1)
        else:
            if tuple(shape) != (5120, BASE_WIDTH // CHANNELS_PER_GROUP):
                raise ValueError(
                    f"unexpected down quantization shape for {name}: {shape}"
                )
            selected = np.take(tensor, group_ids, axis=1)
        new_shape = list(selected.shape)
    payload = selected.tobytes(order="C")
    del tensor, selected
    return new_shape, payload


def write_candidate(
    source: Path,
    draft: Path,
    output: Path,
    scores_path: Path,
    scores: dict,
    source_original_layers: list[int],
    target_taps: list[int],
    widths: list[int],
    source_parameters: int,
    original_base_parameters: int,
    bits: int,
) -> dict:
    layer_count = len(source_original_layers)
    if len(widths) != layer_count or any(
        width <= 0 or width > BASE_WIDTH or width % 256 for width in widths
    ):
        raise ValueError(
            f"each of the {layer_count} widths must be a positive multiple of 256 up to {BASE_WIDTH}"
        )
    uniform = len(set(widths)) == 1
    profile_tag = hashlib.sha256(",".join(map(str, widths)).encode()).hexdigest()[:8]
    name = (
        f"qwen38-{layer_count}l-mlp{widths[0]}"
        if uniform
        else f"qwen38-{layer_count}l-mlpvar-{sum(widths)}-{min(widths)}-{profile_tag}"
    )
    final = output / name
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty candidate: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=output))
    try:
        group_rows: dict[int, np.ndarray] = {}
        group_words: dict[int, np.ndarray] = {}
        retained_groups: dict[int, list[int]] = {}
        scores_by_original = {int(layer["layer"]): layer for layer in scores["layers"]}
        if set(scores_by_original) != set(range(ORIGINAL_LAYERS)):
            raise ValueError(
                "scores must contain each of the 64 original layers exactly once"
            )
        if len(source_original_layers) == 0 or len(set(source_original_layers)) != len(
            source_original_layers
        ):
            raise ValueError(
                "source original-layer mapping must be non-empty and unique"
            )
        for index, original_layer in enumerate(source_original_layers):
            layer = scores_by_original[original_layer]
            values = np.asarray(layer["group_scores"], dtype=np.float64)
            if values.shape != (BASE_WIDTH // CHANNELS_PER_GROUP,):
                raise ValueError(
                    f"invalid score vector for original layer {original_layer}: {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"non-finite channel scores in original layer {original_layer}"
                )
            group_count = widths[index] // CHANNELS_PER_GROUP
            chosen = np.sort(np.argsort(values, kind="stable")[-group_count:])
            group_ids = chosen.astype(np.int64)
            retained_groups[index] = [int(value) for value in group_ids]
            group_rows[index] = (
                group_ids[:, None] * CHANNELS_PER_GROUP
                + np.arange(CHANNELS_PER_GROUP, dtype=np.int64)
            ).reshape(-1)
            group_words[index] = (
                group_ids[:, None] * (CHANNELS_PER_GROUP // PACKED_VALUES_PER_WORD)
                + np.arange(
                    CHANNELS_PER_GROUP // PACKED_VALUES_PER_WORD, dtype=np.int64
                )
            ).reshape(-1)
        config = read_json(source / "config.json")
        text = config.get("text_config")
        if not isinstance(text, dict) or text.get("num_hidden_layers") != layer_count:
            raise ValueError(
                f"source layer count does not match its {layer_count}-layer mapping"
            )
        if text.get("intermediate_size") != BASE_WIDTH:
            raise ValueError(f"source MLP width must be {BASE_WIDTH}")
        quantization = (
            config.get("quantization") or config.get("quantization_config") or {}
        )
        if quantization.get("group_size") != CHANNELS_PER_GROUP:
            raise ValueError("source must use MLX Q4 groups of 64")
        configured_taps = config.get("splash_target_layer_ids", list(BASE_TAPS))
        if list(configured_taps) != target_taps:
            raise ValueError(
                "source target tap IDs do not match the requested DFlash taps"
            )
        text["intermediate_size"] = max(widths)
        config.pop("splash_layer_intermediate_sizes", None)
        if not uniform:
            config["splash_layer_intermediate_sizes"] = widths
        config["language_model_only"] = True
        write_json(staging / "config.json", config)

        draft_config = read_json(draft / "config.json")
        draft_values = draft_config.get("dflash_config", {}).get("target_layer_ids")
        if list(draft_values or []) != target_taps:
            raise ValueError("draft target tap IDs do not match the source model")
        draft_dir = staging / "draft"
        draft_dir.mkdir()
        write_json(draft_dir / "config.json", draft_config)
        draft_files = [
            p for p in draft.iterdir() if p.is_file() and p.suffix == ".safetensors"
        ]
        if len(draft_files) != 1:
            raise ValueError("expected one DFlash safetensors file")
        (draft_dir / draft_files[0].name).symlink_to(draft_files[0].resolve())

        for filename in TOKENIZER_FILES:
            original = source / filename
            if original.is_file():
                (staging / filename).symlink_to(original.resolve())

        weight_map: dict[str, str] = {}
        total_tensor_bytes = 0
        source_files = source_shards(source)
        for shard_number, shard in enumerate(source_files, 1):
            data_start, header = read_safetensors_header(shard)
            output_name = (
                f"model-{shard_number:05d}-of-{len(source_files):05d}.safetensors"
            )
            output_header = {
                "__metadata__": header.get("__metadata__", {"format": "mlx"})
            }
            output_path = staging / output_name
            offset = 0
            for tensor_name, entry in header.items():
                if tensor_name == "__metadata__":
                    continue
                projection = layer_projection(tensor_name)
                if projection is not None and projection[0] not in retained_groups:
                    raise ValueError(f"unexpected layer index in {tensor_name}")
                shape = selected_shape(tensor_name, entry["shape"], widths)
                dtype = DTYPES.get(entry["dtype"])
                if dtype is None:
                    raise ValueError(
                        f"unsupported safetensors dtype {entry['dtype']} for {tensor_name}"
                    )
                byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                output_header[tensor_name] = {
                    "dtype": entry["dtype"],
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_count],
                }
                offset += byte_count
            encoded = json.dumps(
                output_header, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            encoded += b" " * ((-len(encoded)) % 8)
            planned_data_bytes = offset
            offset = 0
            with shard.open("rb") as source_file, output_path.open("wb") as output_file:
                output_file.write(struct.pack("<Q", len(encoded)))
                output_file.write(encoded)
                mapped = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    for tensor_name, entry in header.items():
                        if tensor_name == "__metadata__":
                            continue
                        projection = layer_projection(tensor_name)
                        transformed = None
                        if projection is not None:
                            index = projection[0]
                            transformed = selected_tensor(
                                tensor_name,
                                entry,
                                mapped,
                                data_start,
                                group_rows[index],
                                group_words[index],
                                np.asarray(retained_groups[index], dtype=np.int64),
                                widths,
                            )
                        if transformed is None:
                            start, end = entry["data_offsets"]
                            byte_count = end - start
                            source_file.seek(data_start + start)
                            remaining = byte_count
                            while remaining:
                                block = source_file.read(
                                    min(16 * 1024 * 1024, remaining)
                                )
                                if not block:
                                    raise ValueError(
                                        f"truncated tensor data in {shard.name}"
                                    )
                                output_file.write(block)
                                remaining -= len(block)
                        else:
                            new_shape, payload = transformed
                            byte_count = len(payload)
                            if new_shape != output_header[tensor_name]["shape"]:
                                raise ValueError(
                                    f"pruned shape mismatch for {tensor_name}"
                                )
                            output_file.write(payload)
                            del payload
                        offset += byte_count
                        total_tensor_bytes += byte_count
                        weight_map[tensor_name] = output_name
                finally:
                    mapped.close()
            if offset != planned_data_bytes:
                raise ValueError(f"tensor byte mismatch while writing {shard.name}")
            if output_path.stat().st_size != 8 + len(encoded) + planned_data_bytes:
                raise ValueError(
                    f"safetensors size mismatch while writing {shard.name}"
                )

        write_json(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": total_tensor_bytes}, "weight_map": weight_map},
        )
        parameters = logical_parameter_count(staging, set(range(layer_count)), bits)
        manifest = {
            "method": "calibrated LayerTrim-style MLP channel ranking; exact Q4 group slicing; no requantization, distillation, or fine-tuning",
            "source": str(source.resolve()),
            "draft_source": str(draft.resolve()),
            "calibration_scores": str(scores_path.resolve()),
            "calibration_tokens": scores["calibration_tokens"],
            "calibration_tokens_by_category": scores["calibration_tokens_by_category"],
            "selection_score": "equal mean of per-layer, unit-sum coding and agentic category scores",
            "width_allocation": (
                "uniform width per layer"
                if uniform
                else "global minimum-loss removal of 256-channel bundles using normalized balanced channel scores; DFlash tap layers remain full width"
            ),
            "quantization_bits": bits,
            "group_size": CHANNELS_PER_GROUP,
            "original_layer_count": ORIGINAL_LAYERS,
            "layer_count": layer_count,
            "source_original_layers": source_original_layers,
            "intermediate_size": max(widths),
            "intermediate_sizes": widths,
            "average_intermediate_size": sum(widths) / len(widths),
            "retained_groups_by_layer": retained_groups,
            "dflash_taps": target_taps,
            "logical_text_parameters": parameters,
            "logical_text_parameters_b": round(parameters / 1_000_000_000, 3),
            "input_source_logical_text_parameters": source_parameters,
            "original_base_logical_text_parameters": original_base_parameters,
            "parameter_reduction_from_input_percent": round(
                100 * (source_parameters - parameters) / source_parameters, 2
            ),
            "parameter_reduction_percent": round(
                100
                * (original_base_parameters - parameters)
                / original_base_parameters,
                2,
            ),
            "target_weight_bytes": total_tensor_bytes,
            "target_weight_gib": round(total_tensor_bytes / (1024**3), 3),
            "vision": "omitted; serve with --language-only",
            "candidate_directory": str(final.resolve()),
            "draft_directory": str((final / "draft").resolve()),
        }
        write_json(staging / "pruning_manifest.json", manifest)
        if final.exists():
            final.rmdir()
        staging.rename(final)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def calibrated_width_profile(
    scores: dict,
    source_original_layers: list[int],
    target_taps: list[int],
    target_average_width: int,
    minimum_layer_width: int,
) -> list[int]:
    """Allocate a global MLP width budget by the next 256-channel loss.

    Each layer's group scores are normalized to unit sum by the calibration
    script, so the marginal loss estimates equal-weighted per-layer removal.
    Tapped layers stay full width; this is a ranking heuristic to evaluate,
    not a quality guarantee.
    """
    if target_average_width <= 0:
        raise ValueError("target average width must be positive")
    if minimum_layer_width <= 0 or minimum_layer_width % 256:
        raise ValueError("minimum layer width must be a positive multiple of 256")
    if minimum_layer_width > BASE_WIDTH:
        raise ValueError(f"minimum layer width must not exceed {BASE_WIDTH}")
    layer_count = len(source_original_layers)
    target_total = layer_count * target_average_width
    starting_total = layer_count * BASE_WIDTH
    if target_total > starting_total:
        raise ValueError("target average width exceeds the unpruned width")
    if target_total % 256:
        raise ValueError("target total width must be divisible by 256")
    protected = set(target_taps)
    if any(index < 0 or index >= layer_count for index in protected):
        raise ValueError("a DFlash tap is outside the source target")
    floor_total = len(protected) * BASE_WIDTH + (layer_count - len(protected)) * minimum_layer_width
    if target_total < floor_total:
        raise ValueError(
            f"target budget {target_total} is below the protected-layer floor {floor_total}"
        )

    scores_by_original = {int(layer["layer"]): layer for layer in scores["layers"]}
    ascending_scores: list[np.ndarray] = []
    values_by_layer: list[np.ndarray] = []
    for original_layer in source_original_layers:
        values = np.asarray(scores_by_original[original_layer]["group_scores"], dtype=np.float64)
        if values.shape != (BASE_WIDTH // CHANNELS_PER_GROUP,) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid channel scores for original layer {original_layer}")
        values_by_layer.append(values)
        ascending_scores.append(np.argsort(values, kind="stable"))

    widths = [BASE_WIDTH] * layer_count
    for _ in range((starting_total - target_total) // 256):
        candidates: list[tuple[float, int]] = []
        for layer, width in enumerate(widths):
            if layer in protected or width <= minimum_layer_width:
                continue
            retained_groups = width // CHANNELS_PER_GROUP
            order = ascending_scores[layer]
            first_retained = len(order) - retained_groups
            marginal_groups = order[first_retained : first_retained + 4]
            candidates.append((float(values_by_layer[layer][marginal_groups].sum()), layer))
        if not candidates:
            raise ValueError("the requested width budget cannot be allocated")
        _, selected_layer = min(candidates)
        widths[selected_layer] -= 256
    return widths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--width", action="append", type=int, choices=range(256, BASE_WIDTH, 256)
    )
    parser.add_argument(
        "--target-average-width",
        type=int,
        help="allocate a global per-layer MLP width budget while keeping DFlash tap layers full width",
    )
    parser.add_argument(
        "--minimum-layer-width", type=int, default=10240,
        help="lower bound used by --target-average-width (default: 10240)",
    )
    args = parser.parse_args()
    if args.target_average_width is not None and args.width:
        parser.error("use either --width or --target-average-width, not both")
    source, draft, scores_path, output = (
        path.expanduser().resolve()
        for path in (args.source, args.draft, args.scores, args.output)
    )
    if not source.is_dir() or not draft.is_dir() or not scores_path.is_file():
        parser.error("source, draft, and scores must be existing local paths")
    if any(
        path == source
        or path.is_relative_to(source)
        or path == draft
        or path.is_relative_to(draft)
        for path in (output,)
    ):
        parser.error("output must be outside both source checkpoints")
    scores = read_json(scores_path)
    if (
        scores.get("original_layers") != ORIGINAL_LAYERS
        or scores.get("intermediate_size") != BASE_WIDTH
    ):
        parser.error(
            "scores must describe the original 64-layer Qwen3.8 width-17408 model"
        )
    categories = scores.get("calibration_tokens_by_category", {})
    if not {"coding", "agentic"}.issubset(categories):
        parser.error(
            "calibration scores must include both coding and agentic categories"
        )
    source_config = read_json(source / "config.json")
    text = source_config.get("text_config")
    if not isinstance(text, dict):
        parser.error("source config has no text_config")
    source_layers = text.get("num_hidden_layers")
    if not isinstance(source_layers, int) or not 1 <= source_layers <= ORIGINAL_LAYERS:
        parser.error("source must have between 1 and 64 Qwen3.8 layers")
    layer_manifest_path = source / "pruning_manifest.json"
    if layer_manifest_path.is_file():
        layer_manifest = read_json(layer_manifest_path)
        source_original_layers = layer_manifest.get("retained_original_layers")
        if not isinstance(source_original_layers, list):
            parser.error("source pruning manifest has no retained original layer map")
    elif source_layers == ORIGINAL_LAYERS:
        source_original_layers = list(range(ORIGINAL_LAYERS))
    else:
        parser.error(
            "a layer-pruned source needs pruning_manifest.json for score mapping"
        )
    if (
        len(source_original_layers) != source_layers
        or any(
            not isinstance(index, int) or index < 0 or index >= ORIGINAL_LAYERS
            for index in source_original_layers
        )
        or len(set(source_original_layers)) != len(source_original_layers)
    ):
        parser.error("source original layer map is invalid")
    target_taps = source_config.get("splash_target_layer_ids", list(BASE_TAPS))
    if (
        not isinstance(target_taps, list)
        or len(target_taps) != len(BASE_TAPS)
        or any(
            not isinstance(index, int) or index < 0 or index >= source_layers
            for index in target_taps
        )
    ):
        parser.error("source target layer IDs are invalid")
    quantization = (
        source_config.get("quantization")
        or source_config.get("quantization_config")
        or {}
    )
    if quantization.get("mode", "affine") != "affine" or quantization.get("bits") != 4:
        parser.error("source must use MLX affine Q4")
    source_parameters = logical_parameter_count(source, set(range(source_layers)), 4)
    original_model = Path(scores["source_model"]).expanduser().resolve()
    if not original_model.is_dir():
        parser.error(
            "the original model referenced by calibration scores is unavailable"
        )
    original_base_parameters = logical_parameter_count(
        original_model, set(range(ORIGINAL_LAYERS)), 4
    )
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    default_widths = (
        [10240, 9216] if source_layers == ORIGINAL_LAYERS else [14336, 14080]
    )
    if args.target_average_width is not None:
        try:
            widths = calibrated_width_profile(
                scores,
                source_original_layers,
                target_taps,
                args.target_average_width,
                args.minimum_layer_width,
            )
        except ValueError as error:
            parser.error(str(error))
        candidate_widths = [widths]
    else:
        candidate_widths = [[width] * source_layers for width in (args.width or default_widths)]
    for widths in candidate_widths:
        manifest = write_candidate(
            source,
            draft,
            output,
            scores_path,
            scores,
            source_original_layers,
            target_taps,
            widths,
            source_parameters,
            original_base_parameters,
            4,
        )
        summaries.append(manifest)
        width_label = (
            f"width {widths[0]}"
            if len(set(widths)) == 1
            else f"width profile min={min(widths)} max={max(widths)} avg={sum(widths)/len(widths):.1f}"
        )
        print(
            f"{manifest['candidate_directory']}: {width_label}, "
            f"{manifest['logical_text_parameters_b']:.3f}B params, "
            f"{manifest['target_weight_gib']:.2f} GiB target, "
            f"{manifest['parameter_reduction_percent']:.1f}% below base",
            flush=True,
        )
    write_json(output / "mlp-candidates.json", {"candidates": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
