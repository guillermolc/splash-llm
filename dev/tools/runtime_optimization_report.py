#!/usr/bin/env python3
"""Compare native runtime measurements and check the small inference fixtures.

Example: python dev/tools/runtime_optimization_report.py matrix.jsonl \
  --cases cases.json --tokenizer /path/to/local/tokenizer \
  --reference baseline-original.jsonl --json-output report.json \
  --markdown-output report.md

Warmups are excluded. Latency ratios compare matching case/trial/parameters;
they do not imply significance. Task checks are small fixtures, not a coding
benchmark. Sampled output is checked for correctness without token equality.
"""

import argparse
import ast
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
from collections import defaultdict


COUNTERS = (
    "decode_commands", "lookup_cycles", "lookup_proposals", "lookup_accepted",
    "neural_draft_cycles", "target_only_cycles", "gpu_controlled_cycles",
    "grouped_gpu_commands", "replay_hits", "replay_builds", "replay_bypasses",
)
RESOURCES = (
    "metal_peak_allocated_bytes", "device_allocated_bytes",
    "physical_footprint_peak_bytes", "replay_cached_bytes",
)


def load_records(path):
    records = []
    for number, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
    return records


def key(row):
    return tuple(row[field] for field in (
        "case", "trial", "prompt_tokens", "maximum_new_tokens", "temperature", "seed"
    )) + (row.get("prompt_token_sha256"),)


def measured(records):
    return [row for row in records if row.get("type") == "result" and not row.get("warmup")]


def baselines(records):
    result = {}
    for row in measured(records):
        if row["mode"] != "baseline":
            continue
        identity = key(row)
        if identity in result:
            raise ValueError(f"duplicate baseline for {row['case']} trial {row['trial']}")
        result[identity] = row
    return result


def comparison(row, baseline):
    if baseline is None:
        return {"matched": False, "greedy_equal": None}
    greedy = row["temperature"] == 0
    return {
        "matched": True,
        "greedy_equal": row["output_ids"] == baseline["output_ids"] if greedy else None,
        "decode_latency_speedup": baseline["decode_wall_seconds"] / row["decode_wall_seconds"],
        "end_to_end_latency_speedup": baseline["wall_seconds"] / row["wall_seconds"],
    }


def strip_fences(text):
    text = text.strip()
    fenced = re.fullmatch(r"(```|~~~)[\w+-]*[ \t]*\r?\n(.*?)\r?\n\1", text, re.DOTALL)
    return fenced.group(2).strip() if fenced else text


FUNCTION_RUNNER = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
try:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except (ValueError, OSError):
    pass  # Darwin may refuse lowering this limit; CPU/wall limits remain.
allowed = {name: getattr(__builtins__, name) for name in (
    "dict", "set", "list", "tuple", "len", "iter", "enumerate", "range",
    "bool", "next", "zip", "sorted", "int", "str", "hash", "isinstance"
)}
namespace = {"__builtins__": allowed}
try:
    exec(compile(sys.stdin.read(), "<generated-function>", "exec"), namespace)
    function = namespace["unique_stable"]
    examples = [([], []), ([3, 1, 3, 2, 1], [3, 1, 2]),
                (["b", "a", "b"], ["b", "a"]),
                ([None, "a", None], [None, "a"]),
                ([(1, 2), (1, 2), (2, 3)], [(1, 2), (2, 3)]),
                ([1, True, 2, False, 0], [1, 2, False])]
    for values, expected in examples:
        if function(values) != expected:
            raise ValueError("unique_stable example returned an incorrect result")
    print(json.dumps({"passed": True, "detail": "six unique_stable examples passed"}))
except Exception as error:
    print(json.dumps({"passed": False, "detail": type(error).__name__ + ": " + str(error)}))
'''


def check_function(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return {"passed": False, "detail": f"invalid Python: {error.msg}"}
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "unique_stable" or tree.body[0].decorator_list):
        return {"passed": False, "detail": "expected one undecorated unique_stable function"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return {"passed": None, "detail": "fixture runner does not support imports or global state"}
        if (isinstance(node, ast.Attribute) and node.attr.startswith("_") or
                isinstance(node, ast.Name) and node.id.startswith("__")):
            return {"passed": None, "detail": "fixture runner does not support private attributes"}
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", FUNCTION_RUNNER], input=source,
            text=True, capture_output=True, timeout=3, check=False,
        )
        if result.returncode:
            return {"passed": False, "detail": f"fixture subprocess exited {result.returncode}"}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "fixture exceeded its execution limit"}


def quality(case, decoded):
    if decoded is None:
        return {"checked": False, "passed": None, "detail": "no local tokenizer supplied"}
    text = strip_fences(decoded)
    name = case["name"]
    if name in ("copy_file", "sampled_copy", "rename_file"):
        prompt = next(message["content"] for message in reversed(case["messages"])
                      if message["role"] == "user")
        expected = prompt.split("\n\n", 1)[1].strip()
        if name == "rename_file":
            expected = expected.replace("normalize_city", "clean_city")
        passed = text == expected
        return {"checked": True, "passed": passed, "check": "exact_source",
                "detail": "source matches" if passed else "source differs after outer fences/whitespace removal"}
    if name == "fresh_code":
        result = check_function(text)
        return {"checked": result["passed"] is not None, "check": "unique_stable_examples", **result}
    if name in ("reasoning", "long_context"):
        expected = "869" if name == "reasoning" else "item_0173"
        return {"checked": True, "passed": text == expected, "check": "exact_answer",
                "expected": expected, "actual": text}
    return {"checked": False, "passed": None, "detail": "no fixture checker for this case"}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["mode"]].append(row)
    summaries = []
    for (case, mode), group in sorted(groups.items()):
        item = {"case": case, "mode": mode, "samples": len(group)}
        for field in ("decode_wall_seconds", "wall_seconds", "decode_tokens_per_second"):
            item[f"median_{field}"] = statistics.median(row[field] for row in group)
        for label in ("baseline", "reference"):
            comparisons = [row[label] for row in group if row[label]["matched"]]
            equal = [row["greedy_equal"] for row in comparisons if row["greedy_equal"] is not None]
            item[label] = {"matched": len(comparisons), "greedy_checked": len(equal),
                           "greedy_equal": sum(equal)}
            for metric in ("decode_latency_speedup", "end_to_end_latency_speedup"):
                item[label][f"median_{metric}"] = (
                    statistics.median(row[metric] for row in comparisons) if comparisons else None
                )
        checks = [row["quality"] for row in group if row["quality"]["checked"]]
        item["quality"] = {"checked": len(checks), "passed": sum(row["passed"] for row in checks)}
        item["counters"] = {field: sum(row["counters"][field] for row in group) for field in COUNTERS}
        accepted = sum(row["accepted_tokens"] for row in group)
        drafted = sum(row["drafted_tokens"] for row in group)
        item["acceptance"] = {"accepted": accepted, "drafted": drafted,
                               "fraction": accepted / drafted if drafted else None}
        item["target_row_cycles"] = [sum(row["target_row_cycles"][index] for row in group) for index in range(4)]
        item["resources"] = {field: max(row["resources"][field] for row in group) for field in RESOURCES}
        item["resources"]["swap_increase_bytes"] = max(row["resources"]["swap_increase_bytes"] for row in group)
        item["resources"]["minimum_host_available_bytes"] = min(row["resources"]["minimum_host_available_bytes"] for row in group)
        summaries.append(item)
    return summaries


def analyze(records, cases, decode=None, reference=None):
    local = baselines(records)
    original = baselines(reference or [])
    metadata = next((row for row in records if row.get("type") == "metadata"), {})
    ref_metadata = next((row for row in reference or [] if row.get("type") == "metadata"), {})
    for source in (metadata, ref_metadata):
        if (source.get("cases_sha256") and cases.get("cases_sha256") and
                source["cases_sha256"] != cases["cases_sha256"]):
            raise ValueError("results use a different case-file fingerprint")
    if (metadata.get("model_manifest") and ref_metadata.get("model_manifest") and
            metadata["model_manifest"] != ref_metadata["model_manifest"]):
        raise ValueError("reference uses a different model manifest")
    fixtures = {case["name"]: case for case in cases["cases"]}
    results = []
    for row in measured(records):
        if row["output_tokens"] != len(row["output_ids"]):
            raise ValueError("output token count disagrees with output_ids")
        for metric in ("decode_wall_seconds", "wall_seconds"):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError(f"invalid {metric}")
        if row["case"] not in fixtures:
            raise ValueError(f"missing case fixture: {row['case']}")
        expected_prompt = fixtures[row["case"]].get("prompt_token_sha256")
        if row.get("prompt_token_sha256") and expected_prompt and row["prompt_token_sha256"] != expected_prompt:
            raise ValueError(f"prompt fingerprint differs from fixture: {row['case']}")
        decoded = decode(row["output_ids"]) if decode else None
        item = {field: row[field] for field in (
            "case", "mode", "trial", "temperature", "output_tokens", "decode_wall_seconds", "wall_seconds"
        )}
        item["decode_tokens_per_second"] = row["output_tokens"] / row["decode_wall_seconds"]
        item["baseline"] = comparison(row, local.get(key(row)))
        item["reference"] = comparison(row, original.get(key(row)))
        item["quality"] = quality(fixtures[row["case"]], decoded)
        if decoded is not None:
            item["decoded_text"] = decoded
        item["accepted_tokens"] = sum(row.get("accepted_counts", []))
        item["drafted_tokens"] = sum(row.get("drafted_counts", []))
        item["counters"] = {field: row.get(field, 0) for field in COUNTERS}
        item["target_row_cycles"] = row.get("target_row_cycles", [0] * 4)
        item["resources"] = {field: row.get(field, 0) for field in RESOURCES}
        item["resources"]["swap_increase_bytes"] = max(0, row.get("swap_after_bytes", 0) - row.get("swap_before_bytes", 0))
        item["resources"]["minimum_host_available_bytes"] = row.get("host_available_bytes", 0)
        results.append(item)
    return {
        "schema_version": 1, "metadata": metadata, "reference_metadata": ref_metadata,
        "provided_cases_sha256": cases.get("cases_sha256"),
        "measured_rows": len(results),
        "excluded_warmup_rows": sum(row.get("type") == "result" and bool(row.get("warmup")) for row in records),
        "notes": [
            "Ratios are paired latency ratios, not statistical significance estimates.",
            "Changes of a few percent may be run noise; repeat alternating arms before accepting a gain.",
            "An external reference is a separate run; matching trial numbers do not control thermal or ordering differences.",
            "Legacy rows lack prompt fingerprints: comparisons assume every run used the same cases.bin; the supplied manifest cannot prove this retroactively.",
            "End-to-end wall time includes prefill and decode; runtime setup is excluded.",
            "Task checks are six small custom fixtures, not HumanEval or MBPP.",
            "Sampled outputs are checked for correctness without requiring token equality.",
            "Sampled checks cover the recorded seeds, not preservation of the entire sampling distribution.",
            "device_allocated_bytes is an end-of-arm sample, not a measured device peak.",
        ],
        "summaries": summarize(results), "rows": results,
    }


def markdown(report):
    lines = ["# Native runtime optimization results", "",
             f"Measured rows: {report['measured_rows']}. Excluded warmups: {report['excluded_warmup_rows']}.", "",
             "Latency ratios above 1 favor the candidate. They use matching case/trial/parameters.", "",
             "| Case | Mode | Samples | Decode tok/s | Decode vs baseline | Total seconds | Total vs baseline | Total vs original |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def ratio(value):
        return f"{value:.3f}x" if value is not None else "n/a"
    for row in report["summaries"]:
        base, ref = row["baseline"], row["reference"]
        lines.append(f"| {row['case']} | {row['mode']} | {row['samples']} | {row['median_decode_tokens_per_second']:.1f} | "
                     f"{ratio(base['median_decode_latency_speedup'])} | {row['median_wall_seconds']:.4f} | "
                     f"{ratio(base['median_end_to_end_latency_speedup'])} | {ratio(ref['median_end_to_end_latency_speedup'])} |")
    lines += ["", "## Correctness and exercised features", "",
              "Token equality is checked only for greedy output. Task checks are independent of equality.", "",
              "| Case | Mode | Greedy equal/base | Greedy equal/original | Task checks passed | Accepted/drafted | Lookup cycles | GPU cycles | Replay hits |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["summaries"]:
        base, ref, check = row["baseline"], row["reference"], row["quality"]
        equality = lambda item: f"{item['greedy_equal']}/{item['greedy_checked']}" if item['greedy_checked'] else "n/a"
        checked = f"{check['passed']}/{check['checked']}" if check["checked"] else "not checked"
        counters, acceptance = row["counters"], row["acceptance"]
        lines.append(f"| {row['case']} | {row['mode']} | {equality(base)} | {equality(ref)} | {checked} | "
                     f"{acceptance['accepted']}/{acceptance['drafted']} | {counters['lookup_cycles']} | "
                     f"{counters['gpu_controlled_cycles']} | {counters['replay_hits']} |")
    lines += ["", "## Failed task checks", ""]
    failures = [row for row in report["rows"] if row["quality"]["checked"] and not row["quality"]["passed"]]
    for row in failures:
        lines.append(f"- {row['case']} / {row['mode']} / trial {row['trial']}: " + json.dumps(row["quality"], ensure_ascii=False))
    if not failures:
        lines.append("No failures among the checks performed.")
    lines += ["", "## Resources", ""]
    for field in (*RESOURCES, "swap_increase_bytes"):
        maximum = max((row["resources"][field] for row in report["rows"]), default=0)
        lines.append(f"- {field}: {maximum} bytes ({maximum / 2**30:.3f} GiB).")
    available = min((row["resources"]["minimum_host_available_bytes"] for row in report["rows"]), default=0)
    lines.append(f"- minimum_host_available_bytes: {available} bytes ({available / 2**30:.3f} GiB).")
    lines += ["", *[f"- {note}" for note in report["notes"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--cases", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--tokenizer", type=pathlib.Path)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    args = parser.parse_args()
    decode = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
        decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)
    report = analyze(load_records(args.results), json.loads(args.cases.read_text()), decode,
                     load_records(args.reference) if args.reference else None)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown_output:
        args.markdown_output.write_text(markdown(report))
    if not args.json_output and not args.markdown_output:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(json.dumps({"measured_rows": report["measured_rows"],
                          "task_checks": sum(row["quality"]["checked"] for row in report["rows"]),
                          "task_failures": sum(row["quality"]["checked"] and not row["quality"]["passed"] for row in report["rows"])}))


if __name__ == "__main__":
    main()
