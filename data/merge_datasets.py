"""
merge_datasets.py — Merge multiple JSONL datasets into one train/val split.

USAGE:
    python data/merge_datasets.py \
        --inputs data/train_train.jsonl data/discord_train.jsonl \
        --output data/merged.jsonl \
        --val-split 0.05
"""

import json
import argparse
import random
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", default="data/merged.jsonl")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    all_records = []
    for path in args.inputs:
        records = load_jsonl(path)
        source = Path(path).stem
        print(f"  {source}: {len(records):,} examples")
        all_records.extend(records)

    print(f"\nTotal before dedup: {len(all_records):,}")

    # Deduplicate by assistant response + user prompt
    seen = set()
    clean = []
    for r in all_records:
        assistant = next((m["content"] for m in r.get("messages", []) if m["role"] == "assistant"), "")
        user = next((m["content"] for m in r.get("messages", []) if m["role"] == "user"), "")
        key = assistant + "|" + user
        if key not in seen:
            seen.add(key)
            clean.append(r)

    print(f"Total after dedup:  {len(clean):,} (removed {len(all_records) - len(clean):,})")

    random.seed(42)
    random.shuffle(clean)

    if args.max_pairs:
        clean = clean[: args.max_pairs]

    split_idx = max(1, int(len(clean) * (1 - args.val_split)))
    train = clean[:split_idx]
    val = clean[split_idx:]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_path = output_path.with_name(output_path.stem + "_train.jsonl")
    val_path = output_path.with_name(output_path.stem + "_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train):,})")
    print(f"  Val:   {val_path}  ({len(val):,})")


if __name__ == "__main__":
    main()
