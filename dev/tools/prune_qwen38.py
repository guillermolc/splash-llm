#!/usr/bin/env python3
"""Build DFlash-compatible, text-only Qwen3.8 layer-pruning candidates.

This is a structural extraction tool, not a pruning-quality method: it drops
whole four-layer groups from an already-quantized MLX checkpoint, compacts the
remaining layer indices, and retargets the DFlash config's tap indices. It does
not calibrate, distill, or fine-tune either checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

STAGING_PATHS: list[Path] = []

BASE_LAYER_COUNT = 64
GROUP_SIZE = 4
ORIGINAL_TAPS = (5, 19, 33, 47, 61)
LAYER_KEY = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)
VISION_PREFIXES = ("vision_tower.", "model.visual.")

# Whole hybrid groups retain Splash's 3x GDN + 1x full-attention cadence.
# Group 0 is a soft protection: the 20-layer candidate deliberately omits it.
CANDIDATES = {
    "qwen38-28l": (0, 1, 2, 4, 8, 11, 15),
    "qwen38-24l": (0, 1, 4, 8, 11, 15),
    "qwen38-20l": (1, 4, 8, 11, 15),
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_safetensors_header(path: Path) -> tuple[int, dict]:
    """Return the data-section offset and header without decoding tensor data."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= 100_000_000:
            raise ValueError(f"invalid safetensors header length: {path}")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid safetensors header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    return 8 + header_length, header


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def is_vision(name: str) -> bool:
    return name.startswith(VISION_PREFIXES)


def layer_number(name: str) -> int | None:
    match = LAYER_KEY.match(name)
    return int(match.group(1)) if match else None


def kept_key(name: str, retained: set[int]) -> bool:
    if is_vision(name):
        return False
    layer = layer_number(name)
    return layer is None or layer in retained


def logical_parameter_count(source: Path, retained: set[int], bits: int) -> int:
    """Count logical text parameters from tensor shapes, not packed bytes."""
    packed_factor = 32 // bits
    count = 0
    for shard in sorted(source.glob("*.safetensors")):
        _, header = read_safetensors_header(shard)
        keys = set(header) - {"__metadata__"}
        for name in keys:
            if not kept_key(name, retained):
                continue
            if name.endswith((".scales", ".biases")):
                # MLX affine scales and offsets encode quantization; they
                # are not parameters of the unquantized architecture.
                continue
            if name.endswith(".weight") and (
                name[:-7] + ".scales" in keys
                and name[:-7] + ".biases" in keys
            ):
                factor = packed_factor
            else:
                factor = 1
            shape = header[name]["shape"]
            elements = 1
            for dimension in shape:
                elements *= dimension
            count += elements * factor
    return count


def source_shards(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"no model.safetensors.index.json in {source}")
    index = read_json(index_path)
    names = sorted(set(index.get("weight_map", {}).values()))
    if not names or any(Path(name).name != name for name in names):
        raise ValueError(f"invalid safetensors shard index in {index_path}")
    paths = [source / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ValueError(f"one or more safetensors shards are missing from {source}")
    return paths


def original_layers(config: dict) -> tuple[dict, int]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("source config has no text_config")
    layers = text.get("num_hidden_layers")
    if layers != BASE_LAYER_COUNT:
        raise ValueError(
            f"expected the 64-layer Qwen3.8-27B checkpoint, found {layers!r} layers"
        )
    if text.get("full_attention_interval") != GROUP_SIZE:
        raise ValueError("unexpected Qwen3.8 hybrid layer interval")
    if len(text.get("layer_types", [])) != BASE_LAYER_COUNT:
        raise ValueError("source config must list all 64 layer_types")
    return text, layers


def draft_taps(config: dict) -> tuple[int, ...]:
    flash = config.get("dflash_config")
    taps = flash.get("target_layer_ids") if isinstance(flash, dict) else None
    if tuple(taps or ()) != ORIGINAL_TAPS:
        raise ValueError(f"expected DFlash taps {ORIGINAL_TAPS}, found {taps!r}")
    return ORIGINAL_TAPS


def retained_layers(groups: tuple[int, ...]) -> list[int]:
    return [group * GROUP_SIZE + offset for group in groups for offset in range(GROUP_SIZE)]


def compact_taps(layers: list[int], taps: tuple[int, ...]) -> list[int]:
    mapping = {original: compact for compact, original in enumerate(layers)}
    try:
        return [mapping[tap] for tap in taps]
    except KeyError as error:
        raise ValueError(
            f"candidate dropped DFlash tap {error.args[0]}; keep all tapped groups"
        ) from None


def write_candidate(
    source: Path,
    draft: Path,
    output: Path,
    candidate_name: str,
    groups: tuple[int, ...],
    source_config: dict,
    source_text: dict,
    taps: tuple[int, ...],
    bits: int,
    base_parameters: int,
) -> dict:
    final_destination = output / candidate_name
    if final_destination.exists() and any(final_destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty candidate: {final_destination}")
    destination = Path(tempfile.mkdtemp(prefix=f".{candidate_name}.", dir=output))
    STAGING_PATHS.append(destination)

    originals = retained_layers(groups)
    retained = set(originals)
    new_taps = compact_taps(originals, taps)
    layer_map = {original: compact for compact, original in enumerate(originals)}

    config = json.loads(json.dumps(source_config))
    text = config["text_config"]
    text["num_hidden_layers"] = len(originals)
    text["layer_types"] = [source_text["layer_types"][i] for i in originals]
    config["splash_target_layer_ids"] = new_taps
    config["language_model_only"] = True
    write_json(destination / "config.json", config)

    draft_config = read_json(draft / "config.json")
    draft_taps(draft_config)
    draft_config["dflash_config"]["target_layer_ids"] = new_taps
    draft_dir = destination / "draft"
    draft_dir.mkdir()
    write_json(draft_dir / "config.json", draft_config)
    draft_weights = [
        path for path in draft.iterdir() if path.is_file() and path.suffix == ".safetensors"
    ]
    if len(draft_weights) != 1:
        raise ValueError("expected one DFlash safetensors file")
    (draft_dir / draft_weights[0].name).symlink_to(draft_weights[0].resolve())

    for name in TOKENIZER_FILES:
        original = source / name
        if original.is_file():
            (destination / name).symlink_to(original.resolve())
    for name in ("config.json", "model.safetensors.index.json"):
        if not (source / name).is_file():
            raise ValueError(f"source checkpoint is missing {name}")

    weight_map: dict[str, str] = {}
    tensor_bytes = 0
    for shard_index, shard in enumerate(source_shards(source), start=1):
        output_name = f"model-{shard_index:05d}-of-00003.safetensors"
        data_start, source_header = read_safetensors_header(shard)
        selected = []
        output_header = {"__metadata__": {"format": "mlx"}}
        shard_offset = 0
        for name, entry in source_header.items():
            if name == "__metadata__" or not kept_key(name, retained):
                continue
            match = LAYER_KEY.match(name)
            if match:
                new_name = (
                    f"language_model.model.layers.{layer_map[int(match.group(1))]}"
                    f".{match.group(2)}"
                )
            else:
                new_name = name
            start, end = entry["data_offsets"]
            byte_count = end - start
            output_header[new_name] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [shard_offset, shard_offset + byte_count],
            }
            selected.append((start, byte_count))
            shard_offset += byte_count
            tensor_bytes += byte_count
            weight_map[new_name] = output_name
        if not selected:
            raise ValueError(f"no tensors retained from source shard {shard.name}")
        encoded_header = json.dumps(
            output_header, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        encoded_header += b" " * ((-len(encoded_header)) % 8)
        output_path = destination / output_name
        with shard.open("rb") as source_file, output_path.open("wb") as output_file:
            output_file.write(struct.pack("<Q", len(encoded_header)))
            output_file.write(encoded_header)
            for start, byte_count in selected:
                source_file.seek(data_start + start)
                remaining = byte_count
                while remaining:
                    block = source_file.read(min(16 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f"truncated tensor data in {shard.name}")
                    output_file.write(block)
                    remaining -= len(block)

    write_json(
        destination / "model.safetensors.index.json",
        {"metadata": {"total_size": tensor_bytes}, "weight_map": weight_map},
    )

    params = logical_parameter_count(source, retained, bits)
    manifest = {
        "method": "whole-four-layer-group extraction; no calibration, distillation, or fine-tuning",
        "source": str(source.resolve()),
        "draft_source": str(draft.resolve()),
        "quantization_bits": bits,
        "original_layer_count": BASE_LAYER_COUNT,
        "retained_original_groups": list(groups),
        "retained_original_layers": originals,
        "layer_count": len(originals),
        "original_dflash_taps": list(taps),
        "remapped_dflash_taps": new_taps,
        "logical_text_parameters": params,
        "logical_text_parameters_b": round(params / 1_000_000_000, 3),
        "source_logical_text_parameters": base_parameters,
        "parameter_reduction_percent": round(
            100 * (base_parameters - params) / base_parameters, 2
        ),
        "target_weight_bytes": tensor_bytes,
        "target_weight_gib": round(tensor_bytes / (1024**3), 3),
        "vision": "omitted; serve with --language-only",
        "candidate_directory": str(final_destination.resolve()),
        "draft_directory": str((final_destination / "draft").resolve()),
    }
    write_json(destination / "pruning_manifest.json", manifest)
    if final_destination.exists():
        final_destination.rmdir()
    destination.rename(final_destination)
    STAGING_PATHS.remove(destination)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="local MLX Qwen3.8 snapshot")
    parser.add_argument("--draft", required=True, type=Path, help="local Qwen3.8 DFlash2 snapshot")
    parser.add_argument("--output", required=True, type=Path, help="output directory outside the source snapshot")
    parser.add_argument("--candidate", action="append", choices=tuple(CANDIDATES),
                        help="candidate(s) to build; defaults to all three")
    args = parser.parse_args(argv)
    source, draft, output = (path.expanduser().resolve() for path in (args.source, args.draft, args.output))
    if not source.is_dir() or not draft.is_dir():
        parser.error("--source and --draft must be existing local directories")
    if output == source or output.is_relative_to(source) or output == draft or output.is_relative_to(draft):
        parser.error("--output must be outside both source snapshots")

    source_config = read_json(source / "config.json")
    source_text, _ = original_layers(source_config)
    draft_config = read_json(draft / "config.json")
    taps = draft_taps(draft_config)
    quantization = source_config.get("quantization") or source_config.get("quantization_config")
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        parser.error("source config must declare affine quantization bits")
    base_parameters = logical_parameter_count(
        source, set(range(BASE_LAYER_COUNT)), bits
    )
    candidates = args.candidate or list(CANDIDATES)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in candidates:
        manifest = write_candidate(
            source, draft, output, name, CANDIDATES[name], source_config,
            source_text, taps, bits, base_parameters,
        )
        summaries.append(manifest)
        print(
            f"{name}: {manifest['layer_count']} layers, "
            f"{manifest['logical_text_parameters_b']:.3f}B logical text params, "
            f"{manifest['parameter_reduction_percent']:.1f}% fewer, "
            f"{manifest['target_weight_gib']:.2f} GiB weights; "
            f"taps {manifest['remapped_dflash_taps']}"
        )
    write_json(output / "candidates.json", {"candidates": summaries})
    print(f"Candidates written to {output}")
    print("Use each candidate's draft directory with --draft-model and pass --language-only.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        for staging in STAGING_PATHS:
            shutil.rmtree(staging, ignore_errors=True)
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
