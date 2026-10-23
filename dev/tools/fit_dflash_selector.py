#!/usr/bin/env python3
"""Fit only DFlash2's selector projection from native, verified captures.

This is a bounded calibration experiment for a frozen native draft backbone.
Gold tokens are never inserted into the natural top-16 candidate lists. Only
accepted prefixes and an unambiguously verified first rejection are labels.
Heldout cached scores select the projection; native heldout decoding must then
confirm any gain because a changed selection changes subsequent draft states.

The full draft is never loaded into MLX. BF16 codebooks remain memory mapped,
and only the rows needed by each batch are widened. The exported checkpoint is
a separate byte-for-byte copy with only hidden_projection.weight replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

HIDDEN = 5120
RANK = 256
TOP_K = 16
QUERY_ROWS = 8
MAX_ROWS = 50_000
COPY_BYTES = 16 * 1024**2
PREFIX = "candidate_selector."
PROJECTION = PREFIX + "hidden_projection.weight"
PREDECESSOR = PREFIX + "predecessor_codebook"
SUCCESSOR = PREFIX + "successor_codebook"


def bf16_float(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite projection cannot be exported")
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").tobytes(order="C")


def finite_bf16(words: np.ndarray) -> bool:
    return not bool(np.any((words & np.uint16(0x7F80)) == np.uint16(0x7F80)))


def integers(value: object, shape: tuple[int, ...], name: str, upper: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    if np.any(array < 0) or np.any(array >= upper):
        raise ValueError(f"out-of-range values in {name}")
    return array


def floats(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values of shape {shape}")
    return array


@dataclass
class Row:
    hidden: np.ndarray
    candidates: np.ndarray
    unary: np.ndarray
    predecessor: int
    label: int  # -1 means the gold token was absent from the natural candidates.
    position: int
    edge: np.ndarray
    proposed: int


class Checkpoint:
    """Read only three tensor ranges from a single BF16 safetensors file."""

    def __init__(self, draft: Path):
        self.source = draft / "model.safetensors"
        self.config_path = draft / "config.json"
        config = json.loads(self.config_path.read_text())
        dflash = config.get("dflash_config", {})
        self.vocabulary = int(config.get("vocab_size", 0))
        if (config.get("hidden_size"), dflash.get("selector_rank"),
                dflash.get("selector_top_k"), dflash.get("block_size")) != (HIDDEN, RANK, TOP_K, QUERY_ROWS):
            raise ValueError("draft does not have the supported DFlash2 5120/256/16/8 geometry")
        if self.vocabulary <= TOP_K:
            raise ValueError("invalid draft vocabulary")
        with self.source.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= 16 * 1024**2:
                raise ValueError("invalid safetensors header length")
            header_bytes = stream.read(length)
            if len(header_bytes) != length:
                raise ValueError("truncated safetensors header")
        self.header = json.loads(header_bytes)
        self.data_offset = 8 + length
        self.header_sha256 = hashlib.sha256(header_bytes).hexdigest()
        self.projection_words = self.tensor(PROJECTION, (RANK, HIDDEN))
        self.before = self.tensor(PREDECESSOR, (self.vocabulary, RANK))
        self.after = self.tensor(SUCCESSOR, (self.vocabulary, RANK))
        # Validate in small chunks, without widening entire codebooks to FP32.
        for name, tensor in ((PROJECTION, self.projection_words),
                             (PREDECESSOR, self.before), (SUCCESSOR, self.after)):
            for start in range(0, tensor.shape[0], 2048):
                if not finite_bf16(tensor[start:start + 2048]):
                    raise ValueError(f"nonfinite BF16 values in {name}")

    def tensor(self, name: str, shape: tuple[int, ...]) -> np.memmap:
        entry = self.header.get(name, {})
        if entry.get("dtype") != "BF16" or entry.get("shape") != list(shape):
            raise ValueError(f"expected BF16 {name} with shape {shape}")
        offsets = entry.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError(f"invalid tensor offsets for {name}")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * 2:
            raise ValueError(f"invalid tensor byte count for {name}")
        if self.data_offset + end > self.source.stat().st_size:
            raise ValueError(f"truncated tensor {name}")
        return np.memmap(self.source, mode="r", dtype="<u2",
                         offset=self.data_offset + start, shape=shape)

    @property
    def projection_offset(self) -> int:
        return self.data_offset + self.header[PROJECTION]["data_offsets"][0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(COPY_BYTES):
            digest.update(block)
    return digest.hexdigest()


def load_rows(manifest_path: Path, checkpoint: Checkpoint) -> tuple[dict[str, list[Row]], list[dict]]:
    vocabulary = checkpoint.vocabulary
    manifest = json.loads(manifest_path.read_text())
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("manifest requires a nonempty prompts list")
    rows: dict[str, list[Row]] = {"train": [], "validation": []}
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_prompt_hashes: set[str] = set()
    checked_drafts: set[Path] = set()
    sources = []
    count = 0
    fingerprints: set[str] = set()
    for prompt in prompts:
        identity, split = prompt.get("id"), prompt.get("split")
        if not isinstance(identity, str) or not identity or identity in seen_ids:
            raise ValueError("prompt IDs must be unique nonempty strings")
        if split not in rows:
            raise ValueError(f"invalid split for {identity}")
        prompt_hash = prompt.get("prompt_sha256")
        if (not isinstance(prompt_hash, str) or len(prompt_hash) != 64
                or any(character not in "0123456789abcdef" for character in prompt_hash)):
            raise ValueError(f"{identity}: manifest requires a lowercase SHA256 of prompt token bytes")
        if prompt_hash in seen_prompt_hashes:
            raise ValueError("duplicate tokenized prompts would contaminate train/validation selection")
        seen_prompt_hashes.add(prompt_hash)
        if prompt.get("prompt") is not None:
            prompt_path = Path(prompt["prompt"])
            if not prompt_path.is_absolute():
                raise ValueError("prompt token paths must be absolute")
            if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != prompt_hash:
                raise ValueError(f"{identity}: prompt token bytes disagree with manifest SHA256")
        capture = Path(prompt["capture"])
        if not capture.is_absolute():
            raise ValueError("capture paths must be absolute")
        capture = capture.resolve(strict=True)
        if capture in seen_paths:
            raise ValueError("the same capture cannot occur twice or in both splits")
        seen_paths.add(capture)
        seen_ids.add(identity)
        has_metadata = False
        summary = None
        cycles = 0
        local_rows = 0
        digest = hashlib.sha256()
        with capture.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if len(line) > 16 * 1024**2:
                    raise ValueError("capture line exceeds the bounded 16 MiB input limit")
                digest.update(line)
                if not line.strip():
                    continue
                if summary is not None:
                    raise ValueError("capture contains records after its terminal summary")
                record = json.loads(line)
                if record.get("kind") == "metadata" or "hidden_size" in record:
                    if has_metadata or cycles:
                        raise ValueError("capture must have one metadata record before its cycles")
                    geometry = (record.get("schema_version"), record.get("hidden_size"),
                                record.get("top_k"), record.get("query_rows"))
                    if geometry != (1, HIDDEN, TOP_K, QUERY_ROWS):
                        raise ValueError(f"unsupported capture metadata: {geometry}")
                    model_root = record.get("model")
                    if not isinstance(model_root, str) or not Path(model_root).is_absolute():
                        raise ValueError("capture model must identify its absolute resolved assembly path")
                    captured_draft = (Path(model_root) / "draft").resolve(strict=True)
                    if captured_draft not in checked_drafts:
                        for captured_file, source_file in (
                            (captured_draft / "model.safetensors", checkpoint.source),
                            (captured_draft / "config.json", checkpoint.config_path),
                        ):
                            if not captured_file.samefile(source_file):
                                if (captured_file.stat().st_size != source_file.stat().st_size
                                        or file_sha256(captured_file) != file_sha256(source_file)):
                                    raise ValueError("--draft differs from the draft used for native captures")
                        checked_drafts.add(captured_draft)
                    captured_prompt_hash = record.get("prompt_sha256")
                    if captured_prompt_hash is not None and captured_prompt_hash != prompt_hash:
                        raise ValueError(f"{identity}: capture prompt SHA256 disagrees with manifest")
                    if (prompt.get("prompt_tokens") is not None
                            and record.get("prompt_tokens") != prompt["prompt_tokens"]):
                        raise ValueError(f"{identity}: capture prompt length disagrees with manifest")
                    if record.get("sampling", False):
                        raise ValueError("this calibration probe requires deterministic native captures")
                    fingerprint = record.get("model_fingerprint")
                    if fingerprint is not None:
                        if not isinstance(fingerprint, str):
                            raise ValueError("invalid capture model fingerprint")
                        fingerprints.add(fingerprint)
                        if len(fingerprints) > 1:
                            raise ValueError("capture suite mixes different native model fingerprints")
                    metadata = record
                    has_metadata = True
                    continue
                if not has_metadata:
                    raise ValueError(f"missing capture metadata: {capture}:{line_number}")
                if record.get("kind") == "summary":
                    if record.get("cycle_count") != cycles:
                        raise ValueError("capture summary cycle_count disagrees with its records")
                    summary = record
                    continue
                if record.get("kind", "cycle") != "cycle":
                    raise ValueError("unrecognized capture record")
                cycles += 1
                accepted, retained = record.get("accepted"), record.get("retained")
                if (type(accepted) is not int or not 0 <= accepted <= 7
                        or type(retained) is not int or not 1 <= retained <= 8):
                    raise ValueError("invalid accepted/retained counts")
                target = record.get("target_tokens")
                if not isinstance(target, list) or len(target) < retained:
                    raise ValueError("target_tokens must contain every retained token")
                integers(target, (len(target),), "target_tokens", vocabulary)
                anchor = record.get("anchor")
                if type(anchor) is not int or not 0 <= anchor < vocabulary:
                    raise ValueError("invalid draft anchor")
                proposed = integers(record.get("proposed"), (7,), "proposed", vocabulary)
                candidates = integers(record.get("candidates"), (7, TOP_K), "candidates", vocabulary)
                if any(len(set(map(int, candidate_row))) != TOP_K for candidate_row in candidates):
                    raise ValueError("natural candidate lists must contain distinct token IDs")
                unary = floats(record.get("unary"), (7, TOP_K), "unary")
                edges = floats(record.get("edges"), (7, TOP_K, TOP_K), "edges")
                hidden = integers(record.get("draft_hidden_bf16"), (QUERY_ROWS, HIDDEN),
                                  "draft_hidden_bf16", 65536).astype(np.uint16)
                if not finite_bf16(hidden):
                    raise ValueError("nonfinite captured draft hidden state")
                passed = min(accepted, retained - 1)
                for position in range(passed):
                    if target[position] != int(proposed[position]):
                        raise ValueError("trusted accepted token disagrees with the proposal")
                output = record.get("output_tokens")
                rejection = (accepted < 7 and retained == accepted + 1
                             and isinstance(output, list) and len(output) == retained
                             and target[accepted] != int(proposed[accepted]))
                trusted = passed + int(rejection)
                for position in range(trusted):
                    previous = anchor if position == 0 else int(target[position - 1])
                    edge_index = 0
                    if position:
                        matches = np.flatnonzero(candidates[position - 1] == previous)
                        if len(matches) != 1:
                            raise ValueError("trusted predecessor absent from prior natural candidates")
                        edge_index = int(matches[0])
                    label_match = np.flatnonzero(candidates[position] == target[position])
                    rows[split].append(Row(
                        hidden[position + 1].copy(), candidates[position].astype(np.uint32),
                        unary[position].copy(), previous,
                        int(label_match[0]) if len(label_match) else -1, position,
                        edges[position, edge_index].copy(), int(proposed[position]),
                    ))
                    count += 1
                    local_rows += 1
                    if count > MAX_ROWS:
                        raise ValueError(f"more than {MAX_ROWS} trusted rows exceeds this bounded probe")
        if not has_metadata or not cycles:
            raise ValueError(f"empty capture: {capture}")
        if summary is None:
            raise ValueError(f"capture has no terminal summary: {capture}")
        sources.append({"id": identity, "split": split, "capture": str(capture),
                        "sha256": digest.hexdigest(), "cycles": cycles,
                        "trusted_rows": local_rows,
                        "model_fingerprint": metadata.get("model_fingerprint"),
                        "native_model": metadata.get("model")})
    for split, values in rows.items():
        if not any(row.label >= 0 for row in values):
            raise ValueError(f"{split} has no trusted rows with naturally covered gold tokens")
    return rows, sources


def half_away(value):
    integer = mx.trunc(value)
    fraction = value - integer
    return integer + mx.where(fraction >= 0.5, 1.0, mx.where(fraction <= -0.5, -1.0, 0.0))


def fake_affine4(master):
    """Match GgufPreparation.cpp quantizeGroup, including BF16 source/metadata."""
    weight = mx.stop_gradient(master.astype(mx.bfloat16).astype(mx.float32))
    groups = weight.reshape(RANK, HIDDEN // 64, 64)
    low, high = mx.min(groups, axis=-1, keepdims=True), mx.maximum(
        mx.max(groups, axis=-1, keepdims=True), 0.0)
    low_edge = mx.abs(low) > mx.abs(high)
    step = mx.maximum((high - low) / 15.0, 1e-7)
    step = mx.where(low_edge, step, -step)
    edge = mx.where(low_edge, low, high)
    zero_code = half_away(edge / step)
    nonzero = zero_code != 0
    step = mx.where(nonzero, edge / mx.where(nonzero, zero_code, 1.0), step)
    bias = mx.where(nonzero, edge, 0.0)
    codes = mx.clip(half_away((groups - bias) / step), 0.0, 15.0)
    scale = step.astype(mx.bfloat16).astype(mx.float32)
    bias = bias.astype(mx.bfloat16).astype(mx.float32)
    restored = (codes * scale + bias).reshape(RANK, HIDDEN)
    return master + mx.stop_gradient(restored - master)


def scores(master, batch):
    hidden, before, after, unary, _ = batch
    projected = hidden @ fake_affine4(master).T
    # Native projection storage is BF16; preserve the FP32-master derivative.
    projected = projected + mx.stop_gradient(projected.astype(mx.bfloat16).astype(mx.float32) - projected)
    return unary + mx.sum((before * projected)[:, None, :] * after, axis=-1)


def loss(master, batch):
    logits = scores(master, batch)
    labels = batch[-1]
    chosen = mx.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1)
    return mx.mean(mx.logsumexp(logits, axis=-1) - chosen)


def make_batch(rows: list[Row], checkpoint: Checkpoint):
    candidates = np.stack([row.candidates for row in rows])
    previous = np.asarray([row.predecessor for row in rows], dtype=np.int32)
    return (
        mx.array(bf16_float(np.stack([row.hidden for row in rows]))),
        mx.array(bf16_float(checkpoint.before[previous])),
        mx.array(bf16_float(checkpoint.after[candidates])),
        mx.array(np.stack([row.unary for row in rows])),
        mx.array(np.asarray([max(row.label, 0) for row in rows], dtype=np.int32)),
    )


def evaluate(master, rows: list[Row], checkpoint: Checkpoint, batch_size: int,
             *, parity: bool = False) -> dict:
    positions = [{"trusted": 0, "covered": 0, "correct": 0, "loss_sum": 0.0} for _ in range(7)]
    parity_count = parity_mismatches = native_path_mismatches = 0
    edge_absolute_sum = edge_absolute_max = 0.0
    margin_min = math.inf
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        logits = scores(master, make_batch(selected, checkpoint))
        mx.eval(logits)
        values = np.asarray(logits)
        for row, value in zip(selected, values):
            position = positions[row.position]
            position["trusted"] += 1
            prediction = int(np.argmax(value))
            if row.label >= 0:
                position["covered"] += 1
                position["correct"] += int(prediction == row.label)
                maximum = float(value.max())
                position["loss_sum"] += maximum + float(np.log(
                    np.exp(value.astype(np.float64) - maximum).sum())) - float(value[row.label])
            if parity:
                edge_error = np.abs((value - row.unary) - row.edge)
                edge_absolute_sum += float(edge_error.sum(dtype=np.float64))
                edge_absolute_max = max(edge_absolute_max, float(edge_error.max()))
                native_scores = row.unary + row.edge
                native_best = int(np.argmax(native_scores))
                ordered = np.sort(native_scores)
                margin = float(ordered[-1] - ordered[-2])
                margin_min = min(margin_min, margin)
                if margin > 1e-3:
                    parity_count += 1
                    parity_mismatches += int(prediction != native_best)
                    native_path_mismatches += int(int(row.candidates[native_best]) != row.proposed)
        del logits, values
    total = sum(position["trusted"] for position in positions)
    covered = sum(position["covered"] for position in positions)
    correct = sum(position["correct"] for position in positions)
    loss_sum = sum(position["loss_sum"] for position in positions)

    def summarize(position: dict) -> dict:
        trusted, covered_here = position["trusted"], position["covered"]
        return {"trusted": trusted, "covered": covered_here,
                "coverage": covered_here / trusted if trusted else None,
                "conditional_accuracy": position["correct"] / covered_here if covered_here else None,
                "conditional_ce": position["loss_sum"] / covered_here if covered_here else None}

    result = {"trusted": total, "covered": covered, "coverage": covered / total if total else None,
              "conditional_accuracy": correct / covered if covered else None,
              "conditional_ce": loss_sum / covered if covered else None,
              "positions": [summarize(position) for position in positions]}
    if parity:
        result["native_parity"] = {
            "rows_with_margin_above_1e-3": parity_count,
            "argmax_mismatches": parity_mismatches,
            "argmax_mismatch_fraction": parity_mismatches / parity_count if parity_count else 0.0,
            "native_proposed_inconsistencies": native_path_mismatches,
            "transition_abs_error_mean": edge_absolute_sum / (total * TOP_K) if total else None,
            "transition_abs_error_max": edge_absolute_max,
            "minimum_native_margin": margin_min if math.isfinite(margin_min) else None,
        }
    return result


def export_checkpoint(checkpoint: Checkpoint, output: Path, replacement: bytes, report: dict) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if len(replacement) != RANK * HIDDEN * 2:
        raise ValueError("incorrect projection replacement byte count")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".selector-fit-", dir=output.parent))
    try:
        with checkpoint.source.open("rb") as source, (temporary / "model.safetensors").open("wb") as target:
            remaining = checkpoint.projection_offset
            while remaining:
                block = source.read(min(remaining, COPY_BYTES))
                if not block:
                    raise ValueError("source checkpoint truncated before projection")
                target.write(block)
                remaining -= len(block)
            target.write(replacement)
            source.seek(checkpoint.projection_offset + len(replacement))
            shutil.copyfileobj(source, target, COPY_BYTES)
        if (temporary / "model.safetensors").stat().st_size != checkpoint.source.stat().st_size:
            raise ValueError("exported checkpoint has changed byte length")
        shutil.copy2(checkpoint.config_path, temporary / "config.json")
        (temporary / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(f"output appeared during fit: {output}")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-parity-mismatch", type=float, default=0.02)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000:
        parser.error("--steps must be between 1 and 2000")
    if not 1 <= args.batch_size <= 128:
        parser.error("--batch-size must be between 1 and 128")
    if not math.isfinite(args.lr) or not 0 < args.lr <= 1e-2:
        parser.error("--lr must be positive and no greater than 1e-2")
    if not 0 <= args.max_parity_mismatch <= 1:
        parser.error("--max-parity-mismatch must lie in [0,1]")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.output.exists():
        parser.error("--output must name a new directory")
    return args


def main() -> None:
    args = arguments()
    mx.set_memory_limit(8 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    checkpoint = Checkpoint(args.draft.resolve(strict=True))
    destination_parent = args.output.absolute().parent
    while not destination_parent.exists():
        destination_parent = destination_parent.parent
    if shutil.disk_usage(destination_parent).free < checkpoint.source.stat().st_size + 256 * 1024**2:
        raise ValueError("insufficient free disk space for a separate complete draft checkpoint")
    rows, sources = load_rows(args.manifest, checkpoint)
    original = bf16_float(checkpoint.projection_words)
    master = mx.array(original, dtype=mx.float32)
    mx.eval(master)
    baseline = {split: evaluate(master, values, checkpoint, args.batch_size, parity=True)
                for split, values in rows.items()}
    print(json.dumps({"event": "baseline", "metrics": baseline}, allow_nan=False), file=sys.stderr, flush=True)
    for split in rows:
        parity = baseline[split]["native_parity"]
        if parity["native_proposed_inconsistencies"]:
            raise ValueError(f"{split}: captured native proposals disagree with their own edge argmax")
        if parity["argmax_mismatch_fraction"] > args.max_parity_mismatch:
            raise ValueError(f"{split}: native parity mismatch exceeds --max-parity-mismatch; refusing training")
    training = [row for row in rows["train"] if row.label >= 0]
    rng = np.random.default_rng(args.seed)
    first_moment, second_moment = mx.zeros_like(master), mx.zeros_like(master)
    value_and_grad = mx.value_and_grad(loss)
    best_loss = baseline["validation"]["conditional_ce"]
    best_projection = original.copy()
    best_step = 0
    history = []
    interval = max(1, min(10, args.steps // 10))
    for step in range(1, args.steps + 1):
        indices = rng.choice(len(training), size=min(args.batch_size, len(training)), replace=False)
        batch = make_batch([training[int(index)] for index in indices], checkpoint)
        value, gradient = value_and_grad(master, batch)
        # Keep this small calibration update bounded without changing frozen weights.
        norm = mx.sqrt(mx.sum(gradient * gradient))
        gradient = gradient * mx.minimum(1.0, 1.0 / mx.maximum(norm, 1e-12))
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        master = master - args.lr * corrected_first / (mx.sqrt(corrected_second) + 1e-8)
        mx.eval(master, first_moment, second_moment, value)
        if not math.isfinite(float(value.item())):
            raise ValueError("nonfinite training loss")
        if step % interval == 0 or step == args.steps:
            validation = evaluate(master, rows["validation"], checkpoint, args.batch_size)
            entry = {"step": step, "batch_loss": float(value.item()), "validation": validation}
            history.append(entry)
            print(json.dumps({"event": "validation", **entry}, allow_nan=False), file=sys.stderr, flush=True)
            if validation["conditional_ce"] < best_loss - 1e-6:
                best_loss = validation["conditional_ce"]
                best_projection = np.asarray(master).copy()
                best_step = step
        del batch, value, gradient, norm, corrected_first, corrected_second
    selected = mx.array(best_projection, dtype=mx.float32)
    final = {split: evaluate(selected, values, checkpoint, args.batch_size)
             for split, values in rows.items()}
    replacement = bf16_bytes(best_projection)
    report = {
        "schema_version": 1,
        "experiment": "native-capture selector hidden_projection-only AF4G64 QAT",
        "source_draft": str(checkpoint.source.parent),
        "source_header_sha256": checkpoint.header_sha256,
        "source_projection_sha256": hashlib.sha256(checkpoint.projection_words.tobytes()).hexdigest(),
        "export_projection_sha256": hashlib.sha256(replacement).hexdigest(),
        "manifest": str(args.manifest.resolve()), "captures": sources,
        "steps": args.steps, "learning_rate": args.lr, "batch_size": args.batch_size,
        "seed": args.seed, "gradient_norm_clip": 1.0,
        "optimizer": "Adam FP32 masters, beta1=.9 beta2=.999 eps=1e-8, no weight decay",
        "selected_step": best_step,
        "fallback_to_original": best_step == 0,
        "selection": "lowest heldout conditional CE; original retained without improvement >1e-6",
        "baseline": baseline, "selected": final, "history": history,
        "memory": {"mlx_limit_bytes": 8 * 1024**3, "mlx_cache_limit_bytes": 256 * 1024**2,
                   "max_trusted_rows": MAX_ROWS, "codebooks": "read-only BF16 mmap, batch gathers only",
                   "mlx_peak_bytes": int(mx.get_peak_memory()),
                   "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                   * (1 if sys.platform == "darwin" else 1024)},
        "scope": {"trained": [PROJECTION], "frozen": "both codebooks and entire draft backbone",
                  "target_loaded": False, "target_modified": False,
                  "natural_candidate_lists_modified": False},
        "limitations": [
            "Conditional metrics exclude gold tokens absent from the natural top-16 candidates.",
            "Only correct-prefix rows and a verified first rejection provide labels; later rows are censored.",
            "Captured states and candidate lists are frozen; changed native selections alter later states.",
            "Heldout native decoding is required to establish an acceptance or speed improvement.",
            "Draft acceptance measures do not establish preservation of target benchmark quality.",
        ],
    }
    export_checkpoint(checkpoint, args.output.absolute(), replacement, report)
    print(json.dumps({"output": str(args.output.absolute()), "selected_step": best_step,
                      "fallback_to_original": best_step == 0,
                      "baseline_validation_ce": baseline["validation"]["conditional_ce"],
                      "selected_validation_ce": final["validation"]["conditional_ce"]}, allow_nan=False))


if __name__ == "__main__":
    main()
