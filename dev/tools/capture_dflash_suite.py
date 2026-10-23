#!/usr/bin/env python3
"""Capture a small, sequential native draft diagnostic suite.

Only the tokenizer is loaded in Python. Native processes load one model at a
time. These local coding/tool prompts are diagnostics, not quality benchmarks.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def memory_free_percent() -> int:
    result = subprocess.run(
        ["/usr/bin/memory_pressure"], capture_output=True, text=True,
        timeout=10, check=True,
    )
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", result.stdout)
    if not match:
        raise RuntimeError("cannot read macOS memory-pressure percentage")
    return int(match.group(1))


def prepare(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False,
    )
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    ids = set()
    prompt_hashes = set()
    prompts = []
    for index, row in enumerate(rows):
        name = row["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in ids:
            raise ValueError("dataset IDs must be unique filename-safe strings")
        ids.add(name)
        # Final assistant boundary includes previous tool results, never the
        # answer about to be predicted. The same prefixes feed both models.
        messages = row["messages"]
        boundary = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        kwargs = {"tools": row["tools"]} if row.get("tools") else {}
        text = tokenizer.apply_chat_template(
            messages[:boundary], tokenize=False, add_generation_prompt=True,
            enable_thinking=False, **kwargs,
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not 1 <= len(tokens) <= 8192:
            raise ValueError(f"{name}: token count outside 1..8192")
        encoded = array.array("I", tokens)
        if encoded.itemsize != 4:
            raise RuntimeError("uint32 token serialization unsupported on this platform")
        if sys.byteorder != "little":
            encoded.byteswap()
        prompt_hash = hashlib.sha256(encoded.tobytes()).hexdigest()
        if prompt_hash in prompt_hashes:
            raise ValueError(f"{name}: duplicate tokenized prompt would contaminate the split")
        prompt_hashes.add(prompt_hash)
        path = args.output / f"{name}.u32"
        path.write_bytes(encoded.tobytes())
        prompts.append({
            "id": name, "category": row.get("category"),
            # Fixed prompt-level split; several positions from one request
            # must never be randomized into both train and validation.
            "split": "train" if index % 2 == 0 else "validation",
            "prompt": str(path.resolve()), "prompt_tokens": len(tokens),
            "prompt_sha256": prompt_hash,
            "capture": str((args.output / f"{name}.jsonl").resolve()),
        })
    return {
        "schema_version": 1, "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "tokenizer": str(args.tokenizer.resolve()),
        "model_root": str(args.model.resolve()) if args.model else None,
        "thinking": False, "cycles": args.cycles, "max_tokens": args.max_tokens,
        "capture_hidden": args.hidden == "yes", "prompts": prompts,
        "purpose": "local draft diagnostic; not target quality parity evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="resolved Splash model root with target/ and draft/")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dev/tools/qwen38_coding_agent_calibration.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/engine-tests/draft-capture")
    parser.add_argument("--metallib", type=Path, default=ROOT / "build/splash.metallib")
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--hidden", choices=("yes", "no"), default="yes")
    parser.add_argument("--min-free-percent", type=int, default=40)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cycles <= 256 or not 1 <= args.max_tokens <= 2048:
        parser.error("cycles must be 1..256 and max-tokens 1..2048")
    if not 10 <= args.min_free_percent <= 95:
        parser.error("min-free-percent must be 10..95")
    if not args.prepare_only and (not args.model or not args.model.is_dir()):
        parser.error("capture requires --model with an existing resolved model root")
    if args.output.exists():
        parser.error("output already exists; choose a new experiment directory")
    args.output.mkdir(parents=True)
    manifest = prepare(args)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(manifest_path)
        return
    for prompt in manifest["prompts"]:
        free = memory_free_percent()
        if free < args.min_free_percent:
            raise RuntimeError(f"memory guard: {free}% free, requires {args.min_free_percent}%")
        command = [
            str(args.binary.resolve()), "--model", str(args.model.resolve()),
            "--metallib", str(args.metallib.resolve()), "--prompt", prompt["prompt"],
            "--output", prompt["capture"], "--cycles", str(args.cycles),
            "--max-tokens", str(args.max_tokens), "--hidden", args.hidden,
        ]
        print(f"{prompt['id']}: {prompt['prompt_tokens']} prompt tokens; memory {free}% free", flush=True)
        with (args.output / f"{prompt['id']}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        with Path(prompt["capture"]).open() as captured:
            metadata = json.loads(captured.readline())
        if metadata.get("prompt_sha256") != prompt["prompt_sha256"]:
            raise RuntimeError(f"{prompt['id']}: capture prompt SHA256 disagrees with manifest")
        prompt["memory_free_percent_before"] = free
        prompt["memory_free_percent_after"] = memory_free_percent()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
