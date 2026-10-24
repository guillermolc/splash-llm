import copy
import unittest

from dev.tools import runtime_optimization_report as report


def row(mode="baseline", tokens=(1,), trial=1, temperature=0, decode=2.0, wall=3.0):
    return {
        "type": "result", "case": "reasoning", "mode": mode, "trial": trial,
        "warmup": False, "prompt_tokens": 55, "maximum_new_tokens": 64,
        "temperature": temperature, "seed": 12345, "output_ids": list(tokens),
        "output_tokens": len(tokens), "decode_wall_seconds": decode,
        "wall_seconds": wall, "accepted_counts": [4], "drafted_counts": [7],
        "lookup_cycles": 1, "lookup_proposals": 7, "lookup_accepted": 4,
        "replay_hits": 2, "target_row_cycles": [0, 0, 0, 1],
        "metal_peak_allocated_bytes": 100, "device_allocated_bytes": 120,
        "physical_footprint_peak_bytes": 150, "replay_cached_bytes": 10,
        "swap_before_bytes": 0, "swap_after_bytes": 8,
    }


CASES = {"cases": [{"name": "reasoning", "messages": []}]}


class RuntimeOptimizationReportTests(unittest.TestCase):
    def test_warmup_excluded_and_matching_trial_used(self):
        warmup = {**row(tokens=(3,)), "warmup": True, "trial": 0}
        records = [warmup, row(tokens=(9,), trial=2), row(), row("lookup", decode=1, wall=2)]
        result = report.analyze(records, CASES)
        self.assertEqual(result["measured_rows"], 3)
        self.assertEqual(result["excluded_warmup_rows"], 1)
        comparison = result["rows"][-1]["baseline"]
        self.assertTrue(comparison["greedy_equal"])
        self.assertEqual(comparison["decode_latency_speedup"], 2)
        self.assertEqual(comparison["end_to_end_latency_speedup"], 1.5)

    def test_sampled_quality_does_not_require_token_equality(self):
        result = report.analyze(
            [row(temperature=0.8), row("lookup", tokens=(2,), temperature=0.8)],
            CASES, lambda _: "869",
        )
        self.assertIsNone(result["rows"][-1]["baseline"]["greedy_equal"])
        self.assertTrue(result["rows"][-1]["quality"]["passed"])

    def test_equal_wrong_baseline_is_still_failed_quality(self):
        result = report.analyze([row(), row("lookup")], CASES, lambda _: "889")
        candidate = result["rows"][-1]
        self.assertTrue(candidate["baseline"]["greedy_equal"])
        self.assertFalse(candidate["quality"]["passed"])
        self.assertEqual(candidate["quality"]["expected"], "869")

    def test_reference_comparison_is_separate(self):
        original = [row(tokens=(7,), decode=4, wall=6)]
        result = report.analyze([row(), row("lookup", decode=1, wall=2)], CASES, reference=original)
        candidate = result["rows"][-1]
        self.assertTrue(candidate["baseline"]["greedy_equal"])
        self.assertFalse(candidate["reference"]["greedy_equal"])
        self.assertEqual(candidate["reference"]["end_to_end_latency_speedup"], 3)

    def test_missing_or_different_parameters_are_not_paired(self):
        candidate = row("lookup", trial=2)
        result = report.analyze([row(), candidate], CASES)
        self.assertFalse(result["rows"][-1]["baseline"]["matched"])
        self.assertIsNone(result["summaries"][-1]["baseline"]["median_decode_latency_speedup"])
        different = {**row("lookup"), "maximum_new_tokens": 65}
        self.assertFalse(report.analyze([row(), different], CASES)["rows"][-1]["baseline"]["matched"])
        base = {**row(), "prompt_token_sha256": "A"}
        changed = {**row("lookup"), "prompt_token_sha256": "B"}
        self.assertFalse(report.analyze([base, changed], CASES)["rows"][-1]["baseline"]["matched"])

    def test_medians_counters_acceptance_and_resource_maxima(self):
        first, second = row("lookup"), row("lookup", trial=2, decode=4, wall=5)
        second["metal_peak_allocated_bytes"] = 200
        first["host_available_bytes"], second["host_available_bytes"] = 90, 80
        result = report.analyze([first, second], CASES)
        summary = result["summaries"][0]
        self.assertEqual(summary["median_decode_wall_seconds"], 3)
        self.assertEqual(summary["median_wall_seconds"], 4)
        self.assertEqual(summary["counters"]["replay_hits"], 4)
        self.assertEqual(summary["acceptance"]["accepted"], 8)
        self.assertEqual(summary["acceptance"]["fraction"], 8 / 14)
        self.assertEqual(summary["resources"]["metal_peak_allocated_bytes"], 200)
        self.assertEqual(summary["resources"]["minimum_host_available_bytes"], 80)
        self.assertEqual(summary["target_row_cycles"], [0, 0, 0, 2])
        self.assertIn("## Resources", report.markdown(result))

    def test_exact_copy_and_rename_strip_only_outer_formatting(self):
        source = "def normalize_city(value):\n    return value\n"
        case = {"name": "copy_file", "messages": [{"role": "user", "content": "Return exactly:\n\n" + source}]}
        self.assertTrue(report.quality(case, "```python\n" + source + "```")["passed"])
        self.assertFalse(report.quality(case, source.replace("return", "yield"))["passed"])
        renamed = {**case, "name": "rename_file"}
        self.assertTrue(report.quality(renamed, source.replace("normalize_city", "clean_city"))["passed"])
        self.assertFalse(report.quality(renamed, source)["passed"])

    def test_fresh_function_examples(self):
        correct = "def unique_stable(values):\n    return list(dict.fromkeys(values))"
        self.assertTrue(report.quality({"name": "fresh_code"}, correct)["passed"])
        wrong = "def unique_stable(values):\n    return []"
        self.assertFalse(report.quality({"name": "fresh_code"}, wrong)["passed"])
        self.assertFalse(report.check_function("def broken(:")["passed"])
        unsupported = "def unique_stable(values):\n    import collections\n    return []"
        self.assertFalse(report.quality({"name": "fresh_code"}, unsupported)["checked"])

    def test_long_context_and_missing_tokenizer(self):
        self.assertTrue(report.quality({"name": "long_context"}, "item_0173")["passed"])
        self.assertFalse(report.quality({"name": "long_context"}, "item_0172")["passed"])
        self.assertFalse(report.quality({"name": "reasoning"}, None)["checked"])

    def test_invalid_rows_duplicate_baselines_and_model_mismatch_rejected(self):
        for field, bad in (("wall_seconds", 0), ("decode_wall_seconds", float("nan")), ("output_tokens", 99)):
            invalid = copy.deepcopy(row())
            invalid[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.analyze([invalid], CASES)
        with self.assertRaises(ValueError):
            report.analyze([row(), row()], CASES)
        with self.assertRaisesRegex(ValueError, "different model"):
            report.analyze([{"type": "metadata", "model_manifest": "A"}, row()], CASES,
                           reference=[{"type": "metadata", "model_manifest": "B"}, row()])
        with self.assertRaisesRegex(ValueError, "case-file fingerprint"):
            report.analyze([{"type": "metadata", "cases_sha256": "A"}, row()],
                           {**CASES, "cases_sha256": "B"})


if __name__ == "__main__":
    unittest.main()
