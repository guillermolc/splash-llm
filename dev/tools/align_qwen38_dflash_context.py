#!/usr/bin/env python3
"""Calibrate DFlash2's context projection for a locally pruned Qwen3.8 target.

The original drafter was trained on hidden states from the unpruned target.
This tool captures corresponding tap activations from the base and pruned MLX
targets, then ridge-fits a small update to the draft's `fc.weight` so the
pruned activations reproduce the base draft-context projection outputs. The
draft backbone and selector remain unchanged. It writes a separate draft
checkpoint; neither input checkpoint is modified.

Requires MLX and mlx-lm on Apple Silicon. The fit is a calibration probe, not a
replacement for training DFlash2 against regenerated responses from the
pruned target.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

BASE_TAPS = (5, 19, 33, 47, 61)
HIDDEN_SIZE = 5120
FC_SHAPE = (HIDDEN_SIZE, len(BASE_TAPS) * HIDDEN_SIZE)
FC_BYTES = int(np.prod(FC_SHAPE)) * 2
COPY_CHUNK_BYTES = 16 * 1024 * 1024


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def tap_layers(model_path: Path, *, base: bool) -> tuple[list[int], list[int]]:
    config = read_json(model_path / "config.json")
    selected = config.get("splash_target_layer_ids", list(BASE_TAPS))
    if not isinstance(selected, list) or len(selected) != len(BASE_TAPS):
        raise ValueError(f"invalid DFlash target taps in {model_path / 'config.json'}")
    if any(not isinstance(layer, int) or layer < 0 for layer in selected):
        raise ValueError("target tap IDs must be non-negative integers")
    if base:
        if selected != list(BASE_TAPS):
            raise ValueError(f"base target taps must be {list(BASE_TAPS)}, found {selected}")
        return selected, list(BASE_TAPS)

    manifest_path = model_path / "pruning_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"pruned target has no mapping manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    retained = manifest.get(
        "retained_original_layers", manifest.get("source_original_layers")
    )
    if not isinstance(retained, list) or any(not isinstance(x, int) for x in retained):
        raise ValueError("pruning manifest has no original-layer mapping")
    if max(selected) >= len(retained):
        raise ValueError("a DFlash tap is outside the pruned target layer map")
    mapped = [retained[layer] for layer in selected]
    if mapped != list(BASE_TAPS):
        raise ValueError(
            f"pruned taps map to original layers {mapped}, expected {list(BASE_TAPS)}"
        )
    return selected, mapped


def split_records(
    records: list[dict], holdout_per_category: int
) -> tuple[list[dict], list[dict]]:
    by_category: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(str(record["category"]), []).append(record)
    train: list[dict] = []
    validation: list[dict] = []
    for category, values in by_category.items():
        if len(values) <= holdout_per_category:
            raise ValueError(
                f"category {category!r} needs more than {holdout_per_category} records"
            )
        validation.extend(values[-holdout_per_category:])
        train.extend(values[:-holdout_per_category])
    if not train or not validation:
        raise ValueError("calibration split produced an empty train or validation set")
    return train, validation


def capture_tapped_hidden(
    model_path: Path,
    records: list[dict],
    selected_layers: list[int],
    *,
    label: str,
    max_tokens: int,
) -> tuple[dict[str, np.ndarray], int]:
    print(f"Loading {label} target: {model_path}", flush=True)
    model, tokenizer = load(str(model_path))
    text_model = model.model
    layers = text_model.layers
    if not layers or max(selected_layers) >= len(layers):
        raise ValueError(f"tap layer is outside the {label} model")
    args = model.args.text_config
    if int(args["hidden_size"]) != HIDDEN_SIZE:
        raise ValueError(f"unexpected {label} hidden size {args['hidden_size']}")

    selected = set(selected_layers)
    layer_ids = {id(layer): index for index, layer in enumerate(layers)}
    decoder_type = type(layers[0])
    original_call = decoder_type.__call__
    current: dict[int, np.ndarray] = {}
    captures: dict[str, np.ndarray] = {}
    token_total = 0

    def capture_decoder(self, x, mask=None, cache=None):
        out = original_call(self, x, mask=mask, cache=cache)
        index = layer_ids.get(id(self))
        if index in selected:
            host_out = out.astype(mx.float32)
            mx.eval(host_out)
            rows = np.asarray(host_out).reshape((-1, HIDDEN_SIZE))
            current[index] = rows.copy()
        return out

    model.eval()
    decoder_type.__call__ = capture_decoder
    try:
        for record in records:
            kwargs = {
                "tokenize": True,
                "add_generation_prompt": False,
                "reasoning_effort": "medium",
            }
            if record.get("tools"):
                kwargs["tools"] = record["tools"]
            token_ids = [
                int(token)
                for token in tokenizer.apply_chat_template(record["messages"], **kwargs)
            ]
            if not token_ids:
                raise ValueError(f"record {record.get('id')!r} tokenized to no input")
            if len(token_ids) > max_tokens:
                raise ValueError(
                    f"record {record.get('id')!r} has {len(token_ids)} tokens, "
                    f"above --max-tokens={max_tokens}"
                )
            current.clear()
            # Calling the text backbone avoids materializing a [tokens, vocab]
            # logits tensor; the DFlash taps are intermediate decoder outputs.
            hidden = text_model(mx.array([token_ids]))
            mx.eval(hidden)
            if set(current) != selected:
                raise RuntimeError(
                    f"captured taps {sorted(current)}, expected {sorted(selected)}"
                )
            rows = [current[layer] for layer in selected_layers]
            lengths = {value.shape[0] for value in rows}
            if lengths != {len(token_ids)}:
                raise RuntimeError("captured target tap rows do not match token count")
            captures[str(record["id"])] = np.concatenate(rows, axis=1)
            token_total += len(token_ids)
            print(
                f"Captured {label} {record['id']}: {len(token_ids)} tokens; "
                f"MLX active {mx.get_active_memory() / 2**30:.2f} GiB",
                flush=True,
            )
            del hidden
            mx.clear_cache()
    finally:
        decoder_type.__call__ = original_call
        del model, text_model, layers, tokenizer
        mx.clear_cache()
        gc.collect()
    return captures, token_total


def bf16_to_float32(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return (words << 16).view("<f4")


def float32_to_bf16(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").tobytes(order="C")


def checkpoint_fc(draft: Path) -> tuple[Path, int, bytes]:
    source = draft / "model.safetensors"
    with source.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header_bytes = stream.read(header_length)
        header = json.loads(header_bytes)
    entry = header.get("fc.weight")
    if not isinstance(entry, dict) or entry.get("shape") != list(FC_SHAPE):
        raise ValueError("draft checkpoint does not have the expected fc.weight shape")
    if entry.get("dtype") != "BF16":
        raise ValueError("draft fc.weight must be BF16")
    start, end = entry["data_offsets"]
    if end - start != FC_BYTES:
        raise ValueError("draft fc.weight byte count does not match its shape")
    return source, 8 + header_length + start, header_bytes


def write_draft_checkpoint(
    source: Path, source_fc_offset: int, output: Path, replacement_fc: bytes
) -> None:
    if len(replacement_fc) != FC_BYTES:
        raise ValueError("replacement fc.weight has an incorrect byte count")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing draft: {output}")
    output.mkdir(parents=True)
    temporary = Path(tempfile.mkdtemp(prefix=".draft-align-", dir=output))
    try:
        output_file = temporary / "model.safetensors"
        fc_end = source_fc_offset + FC_BYTES
        with source.open("rb") as src, output_file.open("wb") as dst:
            before_remaining = source_fc_offset
            while before_remaining:
                block = src.read(min(COPY_CHUNK_BYTES, before_remaining))
                if not block:
                    raise ValueError("truncated DFlash checkpoint before fc.weight")
                dst.write(block)
                before_remaining -= len(block)
            dst.write(replacement_fc)
            src.seek(fc_end)
            while True:
                block = src.read(COPY_CHUNK_BYTES)
                if not block:
                    break
                dst.write(block)
            dst.flush()

        shutil.copy2(source.parent / "config.json", temporary / "config.json")
        (temporary / "model.safetensors").replace(output / "model.safetensors")
        (temporary / "config.json").replace(output / "config.json")
        temporary.rmdir()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(output, ignore_errors=True)
        raise


def relative_rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    numerator = np.sum((predicted - reference) ** 2, dtype=np.float64)
    denominator = np.sum(reference**2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, 1e-30)))


def fit_context_projection(
    features: dict[str, np.ndarray],
    records: list[dict],
    train_records: list[dict],
    validation_records: list[dict],
    old_fc: np.ndarray,
    ridge_ratio: float,
) -> tuple[np.ndarray, dict]:
    def stack(selected: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        pruned = np.concatenate(
            [features["pruned"][str(row["id"])] for row in selected], axis=0
        )
        base = np.concatenate(
            [features["base"][str(row["id"])] for row in selected], axis=0
        )
        return pruned, base

    train_pruned, train_base = stack(train_records)
    validation_pruned, validation_base = stack(validation_records)
    if train_pruned.shape[1] != FC_SHAPE[1] or train_base.shape[1] != FC_SHAPE[1]:
        raise ValueError("captured target features do not match the DFlash fc input")

    train_reference = train_base @ old_fc.T
    train_before = train_pruned @ old_fc.T
    residual = train_reference - train_before
    x = mx.array(train_pruned, dtype=mx.float32)
    gram = mx.matmul(x, x.T)
    gram_host = np.asarray(gram, dtype=np.float32)
    scale = float(np.trace(gram_host) / gram_host.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid captured-context covariance")
    system = gram_host + np.eye(gram_host.shape[0], dtype=np.float32) * (
        scale * ridge_ratio
    )
    # MLX's GPU backend does not implement dense linear solves. Solve the
    # small dual system on CPU, then use Metal for the wide projection update.
    coefficients_host = np.linalg.solve(system, residual)
    coefficients = mx.array(coefficients_host, dtype=mx.float32)
    delta = mx.matmul(coefficients.T, x)
    mx.eval(delta)
    new_fc = old_fc + np.asarray(delta, dtype=np.float32)
    if not np.isfinite(new_fc).all():
        raise ValueError("context-projection fit produced non-finite weights")

    metrics = {}
    for label, x_values, base_values in (
        ("train", train_pruned, train_base),
        ("validation", validation_pruned, validation_base),
    ):
        reference = base_values @ old_fc.T
        before = x_values @ old_fc.T
        after = x_values @ new_fc.T
        metrics[label] = {
            "tokens": int(x_values.shape[0]),
            "relative_rmse_before": relative_rmse(before, reference),
            "relative_rmse_after": relative_rmse(after, reference),
        }
    metrics["fit"] = {
        "train_tokens": int(train_pruned.shape[0]),
        "validation_tokens": int(validation_pruned.shape[0]),
        "ridge_ratio": ridge_ratio,
        "ridge_diagonal": scale * ridge_ratio,
        "relative_fc_weight_delta": float(
            np.linalg.norm(new_fc - old_fc) / max(np.linalg.norm(old_fc), 1e-30)
        ),
    }
    del x, gram, gram_host, system, coefficients_host, coefficients, delta
    mx.clear_cache()
    return new_fc, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="unpruned MLX Qwen3.8 target")
    parser.add_argument("--pruned", required=True, type=Path, help="pruned MLX Qwen3.8 target")
    parser.add_argument("--draft", required=True, type=Path, help="candidate-local DFlash2 checkpoint")
    parser.add_argument("--calibration", required=True, type=Path, help="coding/agent JSONL records")
    parser.add_argument("--output", required=True, type=Path, help="new calibrated draft directory")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--holdout-per-category", type=int, default=1)
    parser.add_argument("--ridge-ratio", type=float, default=1e-3)
    parser.add_argument("--memory-limit-gib", type=int, default=32)
    args = parser.parse_args()

    base = args.base.expanduser().resolve()
    pruned = args.pruned.expanduser().resolve()
    draft = args.draft.expanduser().resolve()
    calibration = args.calibration.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (base, pruned, draft):
        if not path.is_dir():
            parser.error(f"not a local checkpoint directory: {path}")
    if not calibration.is_file():
        parser.error(f"calibration file does not exist: {calibration}")
    if output.exists():
        parser.error(f"output already exists; refusing to overwrite: {output}")
    if args.holdout_per_category < 1 or args.ridge_ratio <= 0:
        parser.error("holdout count and ridge ratio must be positive")
    mx.set_memory_limit(args.memory_limit_gib * 1024**3)

    records = read_records(calibration)
    train_records, validation_records = split_records(
        records, args.holdout_per_category
    )
    base_layers, original_taps = tap_layers(base, base=True)
    pruned_layers, mapped_taps = tap_layers(pruned, base=False)
    if mapped_taps != original_taps:
        raise ValueError("base and pruned target taps do not correspond")
    draft_config = read_json(draft / "config.json")
    draft_taps = draft_config.get("dflash_config", {}).get("target_layer_ids")
    if draft_taps != pruned_layers:
        raise ValueError(
            f"draft taps {draft_taps} do not match pruned target taps {pruned_layers}"
        )

    base_capture, base_tokens = capture_tapped_hidden(
        base,
        records,
        base_layers,
        label="base",
        max_tokens=args.max_tokens,
    )
    pruned_capture, pruned_tokens = capture_tapped_hidden(
        pruned,
        records,
        pruned_layers,
        label="pruned",
        max_tokens=args.max_tokens,
    )
    if base_tokens != pruned_tokens or set(base_capture) != set(pruned_capture):
        raise RuntimeError("base and pruned calibration captures are not aligned")

    checkpoint, fc_offset, _ = checkpoint_fc(draft)
    with checkpoint.open("rb") as stream:
        stream.seek(fc_offset)
        old_fc = bf16_to_float32(stream.read(FC_BYTES)).reshape(FC_SHAPE).copy()
    features = {"base": base_capture, "pruned": pruned_capture}
    new_fc, metrics = fit_context_projection(
        features,
        records,
        train_records,
        validation_records,
        old_fc,
        args.ridge_ratio,
    )
    metrics["captures"] = {
        "base_tokens": base_tokens,
        "pruned_tokens": pruned_tokens,
        "training_records": [str(record["id"]) for record in train_records],
        "validation_records": [str(record["id"]) for record in validation_records],
        "taps_base": base_layers,
        "taps_pruned": pruned_layers,
    }

    payload = float32_to_bf16(new_fc)
    if len(payload) != FC_BYTES:
        raise RuntimeError("BF16 encoding changed the context-projection size")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_draft_checkpoint(checkpoint, fc_offset, output, payload)
    (output / "alignment_report.json").write_text(
        json.dumps(
            {
                "method": "ridge context-projection alignment from pruned target taps to base DFlash2 projection outputs",
                "base_model": str(base),
                "pruned_model": str(pruned),
                "draft_source": str(draft),
                "calibration_file": str(calibration),
                "metrics": metrics,
                "scope": "fc.weight only; draft backbone and candidate selector unchanged",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote calibrated DFlash2 draft to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
