#!/usr/bin/env python3
"""Analyze native DFlash2 JSONL captures without loading a model.

Only the accepted prefix and a demonstrably verified first rejection supply
labels. Verifier rows after the rejection are conditioned on rejected draft
tokens and must never become training examples. Examples reference a captured
BF16 hidden row rather than copying the hidden vector into another file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
HIDDEN_SIZE = 5120
RANK = 256
TOP_K = 16
QUERY_ROWS = 8
DRAFT_POSITIONS = QUERY_ROWS - 1
REPLAY_ATOL = 1e-5
REPLAY_RTOL = 1e-6

CAVEATS = [
    "Only the accepted prefix and a verified first rejection are labeled; "
    "the remaining verifier suffix is censored.",
    "The oracle is limited to unary top-16 candidates at trusted captured "
    "positions. It is an offline selector opportunity, not a prediction of "
    "acceptance after later tokens, hidden states, or distributions change.",
    "Acceptance and selector accuracy do not measure the pruned target's "
    "coding quality or its quality relative to the original target.",
    "Raw GPU acceptance includes seven offered draft positions per cycle; "
    "budget or terminal truncation can leave fewer trusted positions.",
]


def integer(value, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{name} must be an integer {bounds}")
    return value


def finite_number(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def array(value, name: str, length: int | None = None) -> list:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        expected = "an array" if length is None else f"an array of length {length}"
        raise ValueError(f"{name} must be {expected}")
    return value


def tokens(value, name: str, length: int | None = None) -> list[int]:
    result = array(value, name, length)
    for index, token in enumerate(result):
        integer(token, f"{name}[{index}]")
    return result


def required(record: dict, name: str):
    if name not in record:
        raise ValueError(f"missing {name}")
    return record[name]


def validate_metadata(record: dict) -> None:
    for field, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("hidden_size", HIDDEN_SIZE),
        ("rank", RANK),
        ("top_k", TOP_K),
        ("query_rows", QUERY_ROWS),
    ):
        actual = integer(required(record, field), field)
        if actual != expected:
            raise ValueError(f"unsupported {field}={actual}; expected {expected}")
    if not isinstance(required(record, "model"), str) or not record["model"]:
        raise ValueError("model must be a nonempty string")
    integer(required(record, "prompt_tokens"), "prompt_tokens")
    if "prompt_sha256" in record:
        digest = record["prompt_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("prompt_sha256 must be a lowercase SHA256 hex string")
    if "prefill_output_tokens" in record:
        tokens(record["prefill_output_tokens"], "prefill_output_tokens")
    if record.get("prefill_first_token") is not None:
        integer(record["prefill_first_token"], "prefill_first_token")


def validate_cycle(record: dict) -> None:
    for field in ("request_id", "cycle", "position", "anchor"):
        integer(required(record, field), field)
    integer(required(record, "accepted"), "accepted", 0, DRAFT_POSITIONS)
    retained = integer(required(record, "retained"), "retained", 1, QUERY_ROWS)
    candidates = array(required(record, "candidates"), "candidates", DRAFT_POSITIONS)
    unary = array(required(record, "unary"), "unary", DRAFT_POSITIONS)
    edges = array(required(record, "edges"), "edges", DRAFT_POSITIONS)
    proposed = tokens(required(record, "proposed"), "proposed", DRAFT_POSITIONS)
    for position in range(DRAFT_POSITIONS):
        ids = tokens(candidates[position], f"candidates[{position}]", TOP_K)
        if len(set(ids)) != TOP_K:
            raise ValueError(f"candidates[{position}] contains duplicate token IDs")
        if proposed[position] not in ids:
            raise ValueError(f"proposed[{position}] is outside its candidate set")
        row = array(unary[position], f"unary[{position}]", TOP_K)
        for index, score in enumerate(row):
            finite_number(score, f"unary[{position}][{index}]")
        matrix = array(edges[position], f"edges[{position}]", TOP_K)
        for predecessor, values in enumerate(matrix):
            row = array(values, f"edges[{position}][{predecessor}]", TOP_K)
            for index, score in enumerate(row):
                finite_number(score, f"edges[{position}][{predecessor}][{index}]")
    targets = tokens(required(record, "target_tokens"), "target_tokens", retained)
    outputs = tokens(required(record, "output_tokens"), "output_tokens")
    if not outputs or outputs[0] != record["anchor"]:
        raise ValueError("output_tokens must begin with anchor")
    integer(
        required(record, "output_tokens_without_kv"),
        "output_tokens_without_kv",
        0,
        len(outputs),
    )
    if type(required(record, "finished")) is not bool:
        raise ValueError("finished must be a boolean")
    passed = min(record["accepted"], retained - 1)
    if targets[:passed] != proposed[:passed]:
        raise ValueError("accepted trusted prefix disagrees with target_tokens")
    if "draft_hidden_bf16" in record:
        hidden = array(record["draft_hidden_bf16"], "draft_hidden_bf16", QUERY_ROWS)
        for row_index, row in enumerate(hidden):
            values = array(row, f"draft_hidden_bf16[{row_index}]", HIDDEN_SIZE)
            for column, bits in enumerate(values):
                integer(bits, f"draft_hidden_bf16[{row_index}][{column}]", 0, 65535)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def empty_labels() -> dict:
    return {
        "trusted": 0,
        "candidate_hits": 0,
        "natural_candidate_misses": 0,
        "chosen_correct": 0,
        "selector_misses": 0,
    }


def label_report(counts: dict) -> dict:
    return {
        **counts,
        "recall_top16": ratio(counts["candidate_hits"], counts["trusted"]),
        "chosen_accuracy": ratio(counts["chosen_correct"], counts["trusted"]),
        "selector_accuracy_given_candidate_hit": ratio(
            counts["chosen_correct"], counts["candidate_hits"]
        ),
        "selector_miss_rate_given_candidate_hit": ratio(
            counts["selector_misses"], counts["candidate_hits"]
        ),
    }


class Accumulator:
    def __init__(self) -> None:
        self.labels = empty_labels()
        self.by_position = [empty_labels() for _ in range(DRAFT_POSITIONS)]
        self.cycles = 0
        self.raw_accepted = 0
        self.passed = 0
        self.valid_rejections = 0
        self.recoverable_rejections = 0
        self.censored_positions = 0
        self.cycles_with_hidden = 0
        self.examples = 0
        self.accepted_histogram = Counter()
        self.retained_histogram = Counter()
        self.trusted_histogram = Counter()
        self.oracle_prefix_histogram = Counter()
        self.oracle_prefix_tokens = 0
        self.oracle_additional_tokens = 0
        self.replay_exact = 0
        self.replay_ties = 0
        self.replay_mismatches = 0
        self.replay_compatible_cycles = 0
        self.replay_failures = []

    def add_cycle(self, record: dict, path: Path, line: int, examples_stream) -> None:
        accepted, retained = record["accepted"], record["retained"]
        passed = min(accepted, retained - 1)
        # A short terminal emission can make the raw counters ambiguous. Only
        # an emitted complete retained block proves the rejection label here.
        valid_rejection = (
            accepted < DRAFT_POSITIONS
            and retained == accepted + 1
            and len(record["output_tokens"]) == retained
            and record["target_tokens"][accepted] != record["proposed"][accepted]
        )
        trusted = passed + int(valid_rejection)
        replay = replay_native(record)
        self.cycles += 1
        self.raw_accepted += accepted
        self.passed += passed
        self.valid_rejections += int(valid_rejection)
        self.censored_positions += DRAFT_POSITIONS - trusted
        self.cycles_with_hidden += int("draft_hidden_bf16" in record)
        self.accepted_histogram[accepted] += 1
        self.retained_histogram[retained] += 1
        self.trusted_histogram[trusted] += 1
        self.replay_exact += replay["exact"]
        self.replay_ties += replay["ties"]
        self.replay_mismatches += len(replay["mismatches"])
        self.replay_compatible_cycles += int(not replay["mismatches"])
        for failure in replay["mismatches"]:
            if len(self.replay_failures) < 20:
                self.replay_failures.append(
                    {"capture": str(path), "line": line, "cycle": record["cycle"], **failure}
                )
        oracle_prefix = 0
        oracle_open = True
        for position in range(trusted):
            gold = record["target_tokens"][position]
            ids = record["candidates"][position]
            hit = gold in ids
            correct = record["proposed"][position] == gold
            for counts in (self.labels, self.by_position[position]):
                counts["trusted"] += 1
                counts["candidate_hits"] += int(hit)
                counts["natural_candidate_misses"] += int(not hit)
                counts["chosen_correct"] += int(correct)
                counts["selector_misses"] += int(hit and not correct)
            if oracle_open and hit:
                oracle_prefix += 1
            else:
                oracle_open = False
            if valid_rejection and position == accepted and hit:
                self.recoverable_rejections += 1
            # Conditional CE cannot learn to select a token absent from top K.
            # It remains a miss in every metric, and is excluded only here.
            if hit:
                self.examples += 1
                if examples_stream is not None:
                    example = {
                        "capture": str(path),
                        "line": line,
                        "request_id": record["request_id"],
                        "cycle": record["cycle"],
                        "cycle_position": record["position"],
                        "position": position,
                        "row": position + 1,
                        "candidate_ids": ids,
                        "unary": record["unary"][position],
                        "gold_index": ids.index(gold),
                        "gold_token": gold,
                        "predecessor": record["anchor"] if position == 0 else record["target_tokens"][position - 1],
                        "proposed": record["proposed"][position],
                        "trusted": True,
                        "label_source": "verified_rejection" if position == passed else "accepted_prefix",
                        "hidden_available": "draft_hidden_bf16" in record,
                        "native_replay_compatible": not replay["mismatches"],
                    }
                    examples_stream.write(json.dumps(example, allow_nan=False) + "\n")
        self.oracle_prefix_tokens += oracle_prefix
        self.oracle_additional_tokens += max(0, oracle_prefix - passed)
        self.oracle_prefix_histogram[oracle_prefix] += 1

    def report(self) -> dict:
        return {
            "cycles": {
                "count": self.cycles,
                "raw_accepted_draft_tokens": self.raw_accepted,
                "raw_offered_draft_tokens": self.cycles * DRAFT_POSITIONS,
                "raw_acceptance_ratio": ratio(self.raw_accepted, self.cycles * DRAFT_POSITIONS),
                "mean_raw_accepted_draft_tokens": ratio(self.raw_accepted, self.cycles),
                "trusted_passed_tokens": self.passed,
                "verified_rejections": self.valid_rejections,
                "trusted_prefix_acceptance_ratio": ratio(self.passed, self.passed + self.valid_rejections),
                "censored_draft_positions": self.censored_positions,
                "with_hidden_states": self.cycles_with_hidden,
                "accepted_histogram": dict(sorted(self.accepted_histogram.items())),
                "retained_histogram": dict(sorted(self.retained_histogram.items())),
                "trusted_length_histogram": dict(sorted(self.trusted_histogram.items())),
            },
            "labels": label_report(self.labels),
            "by_position": [
                {"position": position, "hidden_row": position + 1, **label_report(counts)}
                for position, counts in enumerate(self.by_position)
            ],
            "oracle": {
                "observed_top16_prefix_tokens": self.oracle_prefix_tokens,
                "additional_trusted_prefix_tokens": self.oracle_additional_tokens,
                "recoverable_verified_rejections": self.recoverable_rejections,
                "recoverable_rejection_ratio": ratio(self.recoverable_rejections, self.valid_rejections),
                "observed_prefix_histogram": dict(sorted(self.oracle_prefix_histogram.items())),
            },
            "native_replay": {
                "compatible": self.replay_mismatches == 0,
                "compatible_cycles": self.replay_compatible_cycles,
                "exact_first_argmax_matches": self.replay_exact,
                "tolerated_float32_ties": self.replay_ties,
                "mismatches": self.replay_mismatches,
                "score_absolute_tolerance": REPLAY_ATOL,
                "score_relative_tolerance": REPLAY_RTOL,
                "first_failures": self.replay_failures,
                "method": "Replay each native transition with first argmax of unary + edge; "
                "condition later rows on the native predecessor so a tolerated tie "
                "does not create spurious downstream mismatches.",
            },
            "conditional_ce_examples": self.examples,
        }


def replay_native(record: dict) -> dict:
    exact, ties, mismatches = 0, 0, []
    previous_index = 0
    for position in range(DRAFT_POSITIONS):
        ids = record["candidates"][position]
        scores = [
            float(unary) + float(edge)
            for unary, edge in zip(
                record["unary"][position], record["edges"][position][previous_index]
            )
        ]
        if any(not math.isfinite(value) for value in scores):
            raise ValueError(f"non-finite unary + edge score at position {position}")
        # max returns the first maximum, matching the native tie convention.
        best = max(range(TOP_K), key=scores.__getitem__)
        native = ids.index(record["proposed"][position])
        if native == best:
            exact += 1
        elif math.isclose(scores[native], scores[best], rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
            ties += 1
        else:
            mismatches.append(
                {
                    "position": position,
                    "native_token": ids[native],
                    "replayed_token": ids[best],
                    "native_score": scores[native],
                    "best_score": scores[best],
                }
            )
        previous_index = native
    return {"exact": exact, "ties": ties, "mismatches": mismatches}


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value}")


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def analyze_capture(path: Path, total: Accumulator, examples_stream) -> dict:
    metadata = None
    summary = None
    local = Accumulator()
    last_cycles = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                if summary is not None:
                    raise ValueError("record follows final summary")
                kind = required(record, "kind")
                if kind == "metadata":
                    if metadata is not None:
                        raise ValueError("duplicate metadata record")
                    validate_metadata(record)
                    metadata = record
                elif metadata is None:
                    raise ValueError("metadata must precede cycle or summary records")
                elif kind == "cycle":
                    validate_cycle(record)
                    request_id, cycle = record["request_id"], record["cycle"]
                    if request_id in last_cycles and cycle <= last_cycles[request_id]:
                        raise ValueError("cycle numbers must increase within each request")
                    last_cycles[request_id] = cycle
                    local.add_cycle(record, path, line_number, None)
                    total.add_cycle(record, path, line_number, examples_stream)
                elif kind == "summary":
                    count = integer(required(record, "cycle_count"), "cycle_count")
                    if count != local.cycles:
                        raise ValueError(f"summary cycle_count={count} differs from {local.cycles} cycle records")
                    summary = record
                else:
                    raise ValueError(f"unsupported record kind {kind!r}")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if metadata is None:
        raise ValueError(f"{path}: missing metadata")
    if summary is None:
        raise ValueError(f"{path}: missing final summary; capture may be incomplete")
    return {
        "capture": str(path),
        "metadata": metadata,
        "has_final_summary": summary is not None,
        "summary": summary,
        **local.report(),
    }


def temporary_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="Native DFlash2 JSONL captures")
    parser.add_argument("--output", required=True, type=Path, help="Aggregate JSON report")
    parser.add_argument("--examples", type=Path, help="Trusted top-16-hit examples for conditional CE")
    args = parser.parse_args(argv)
    inputs = [path.resolve() for path in args.captures]
    output = args.output.resolve()
    examples = args.examples.resolve() if args.examples else None
    if len(set(inputs)) != len(inputs):
        parser.error("capture paths must be distinct")
    if output in inputs or examples in inputs or output == examples:
        parser.error("report and examples must be distinct from each other and all captures")
    example_stream = None
    report_stream = None
    try:
        if examples is not None:
            example_stream = temporary_output(examples)
        total = Accumulator()
        captures = [analyze_capture(path, total, example_stream) for path in inputs]
        report = {
            "schema_version": SCHEMA_VERSION,
            "analysis": "censored_verified_dflash2_selector",
            "caveats": CAVEATS,
            "captures": captures,
            **total.report(),
        }
        report_stream = temporary_output(output)
        json.dump(report, report_stream, indent=2, allow_nan=False)
        report_stream.write("\n")
        report_stream.close()
        if example_stream is not None:
            example_stream.close()
            os.replace(example_stream.name, examples)
        os.replace(report_stream.name, output)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        for stream in (example_stream, report_stream):
            if stream is not None:
                stream.close()
                Path(stream.name).unlink(missing_ok=True)
    labels = report["labels"]
    print(
        f"{total.cycles} cycles; {labels['trusted']} trusted labels; "
        f"{labels['candidate_hits']} top-16 hits; "
        f"{total.recoverable_rejections} recoverable rejections; "
        f"{total.replay_mismatches} replay mismatches. Report: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
