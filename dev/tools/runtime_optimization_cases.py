#!/usr/bin/env python3
"""Create reproducible inference-only cases from a local tokenizer."""

import argparse
import hashlib
import json
import pathlib
import struct

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    module = "\n\n".join(
        f"def normalize_{name}(value):\n"
        '    """Return a stripped lowercase string, or an empty string for None."""\n'
        "    if value is None:\n"
        "        return \"\"\n"
        "    return str(value).strip().lower()"
        for name in ("name", "city", "country", "category")
    ) + "\n"
    records = "\n".join(
        f'    {{"id": {index}, "name": "item_{index:04d}", "enabled": '
        + ("True" if index % 3 else "False")
        + "},"
        for index in range(256)
    )
    cases = [
        ("copy_file", "Return this Python module exactly, without Markdown fences or explanation:\n\n" + module, 768, 0.0),
        ("rename_file", "In this module, rename normalize_city to clean_city. Return the complete updated module, without Markdown fences or explanation:\n\n" + module, 768, 0.0),
        ("fresh_code", "Implement one Python function unique_stable(values) that returns unique hashable values in their original order. Return only the complete function, without Markdown fences or explanation.", 256, 0.0),
        ("reasoning", "A service handles 37 requests each minute for 24 minutes. Exactly 19 requests fail. How many succeed? Return only the integer.", 64, 0.0),
        ("long_context", "Read the following Python data.\nrecords = [\n" + records + "\n]\nReturn only the name of the record with id 173, without quotes or explanation.", 64, 0.0),
        ("sampled_copy", "Return this Python module exactly, without Markdown fences or explanation:\n\n" + module, 768, 0.8),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with args.output.open("wb") as output:
        output.write(struct.pack("<I", len(cases)))
        for name, prompt, maximum, temperature in cases:
            messages = [
                {"role": "system", "content": "You are a precise coding assistant."},
                {"role": "user", "content": prompt},
            ]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(tokens, "keys"):
                tokens = tokens["input_ids"]
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
            encoded_name = name.encode("utf-8")
            output.write(struct.pack("<I", len(encoded_name)))
            output.write(encoded_name)
            output.write(struct.pack("<I", len(tokens)))
            output.write(struct.pack(f"<{len(tokens)}I", *tokens))
            output.write(struct.pack("<IfQ", maximum, temperature, 12345))
            manifest.append({
                "name": name, "messages": messages, "prompt_tokens": len(tokens),
                "prompt_token_sha256": hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest(),
                "maximum_new_tokens": maximum, "temperature": temperature, "seed": 12345,
            })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "schema_version": 1, "tokenizer": str(args.tokenizer.resolve()),
        "cases_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cases": manifest,
    }, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "prompt_lengths": {c["name"]: c["prompt_tokens"] for c in manifest}}))


if __name__ == "__main__":
    main()
