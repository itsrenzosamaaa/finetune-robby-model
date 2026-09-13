"""
analyze_dataset.py — Analyze and report quality of a JSONL training dataset.

Checks for:
- Total examples
- Duplicate responses
- Short responses (potential noise)
- Language distribution (English / Tagalog / Taglish)
- Response length distribution
- Source distribution (if metadata.source present)
- Top most common responses (reveals if model will overfit to one phrase)

USAGE:
    python data/analyze_dataset.py --input data/train_train.jsonl
    python data/analyze_dataset.py --input data/train_train.jsonl --dedup --output data/train_clean.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter


# ── Language detection (simple heuristic) ────────────────────────────────────
TAGALOG_WORDS = {
    "ako", "ka", "siya", "kami", "tayo", "kayo", "sila", "ang", "ng", "sa",
    "na", "at", "ay", "ito", "iyon", "dito", "diyan", "doon", "naman", "nga",
    "din", "rin", "lang", "ba", "kasi", "pero", "yung", "yun", "ano", "sino",
    "bakit", "paano", "kelan", "saan", "maganda", "mahal", "gusto", "ayaw",
    "kumusta", "salamat", "oo", "hindi", "wala", "may", "meron", "talaga",
    "haha", "hehe", "bro", "pre", "tol", "pare", "boss", "kuya", "ate",
    "grabe", "sige", "ganun", "ganon", "parang", "medyo", "sobrang", "super",
    "yung", "yun", "pala", "kaya", "kasi", "diba", "noh", "ah", "eh",
    "amo", "basta", "aye", "oo", "naa", "wala", "lagi", "tapos",
}

def detect_language(text: str) -> str:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    tagalog_hits = len(words & TAGALOG_WORDS)
    total_words = len(words)
    if total_words == 0:
        return "unknown"
    ratio = tagalog_hits / total_words
    if ratio > 0.4:
        return "tagalog"
    elif ratio > 0.15:
        return "taglish"
    else:
        return "english"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  SKIP line {i+1}: {e}")
    return records


def get_assistant_text(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def get_user_text(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def analyze(records: list[dict]) -> dict:
    total = len(records)
    responses = [get_assistant_text(r) for r in records]
    prompts = [get_user_text(r) for r in records]

    # Duplicates
    response_counter = Counter(responses)
    duplicate_responses = {k: v for k, v in response_counter.items() if v > 1}
    n_duplicates = sum(v - 1 for v in duplicate_responses.values())

    # Short responses (< 5 chars)
    short_responses = [r for r in responses if len(r.strip()) < 5]

    # Length distribution
    lengths = [len(r.split()) for r in responses]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    short_count = sum(1 for l in lengths if l <= 2)
    medium_count = sum(1 for l in lengths if 3 <= l <= 10)
    long_count = sum(1 for l in lengths if l > 10)

    # Language distribution
    lang_counts = Counter(detect_language(r) for r in responses)

    # Source distribution
    sources = [r.get("metadata", {}).get("source", "unknown") for r in records]
    source_counts = Counter(sources)

    # Top most common responses (overfitting risk)
    top_responses = response_counter.most_common(15)

    return {
        "total": total,
        "n_duplicates": n_duplicates,
        "n_unique_responses": len(set(responses)),
        "n_short_responses": len(short_responses),
        "short_examples": short_responses[:5],
        "avg_response_words": round(avg_len, 1),
        "length_distribution": {
            "short (1-2 words)": short_count,
            "medium (3-10 words)": medium_count,
            "long (10+ words)": long_count,
        },
        "language_distribution": dict(lang_counts),
        "source_distribution": dict(source_counts),
        "top_responses": top_responses,
    }


def print_report(stats: dict):
    print("\n" + "="*60)
    print("  DATASET ANALYSIS REPORT")
    print("="*60)

    print(f"\n📊 OVERVIEW")
    print(f"  Total examples     : {stats['total']:,}")
    print(f"  Unique responses   : {stats['n_unique_responses']:,}")
    print(f"  Duplicate pairs    : {stats['n_duplicates']:,}")
    print(f"  Short responses    : {stats['n_short_responses']:,} (< 5 chars)")
    print(f"  Avg response words : {stats['avg_response_words']}")

    print(f"\n📏 RESPONSE LENGTH")
    for label, count in stats["length_distribution"].items():
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<22} {count:>6,} ({pct:5.1f}%) {bar}")

    print(f"\n🌐 LANGUAGE")
    for lang, count in sorted(stats["language_distribution"].items(), key=lambda x: -x[1]):
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {lang:<12} {count:>6,} ({pct:5.1f}%) {bar}")

    if len(stats["source_distribution"]) > 1:
        print(f"\n📁 SOURCE")
        for src, count in sorted(stats["source_distribution"].items(), key=lambda x: -x[1]):
            pct = count / stats["total"] * 100
            print(f"  {src:<16} {count:>6,} ({pct:5.1f}%)")

    print(f"\n⚠️  TOP REPEATED RESPONSES (overfitting risk if too high)")
    for response, count in stats["top_responses"]:
        preview = response[:60].replace("\n", " ")
        if count > 1:
            print(f"  x{count:<4} \"{preview}\"")

    if stats["short_examples"]:
        print(f"\n🗑️  SAMPLE SHORT RESPONSES (consider filtering)")
        for ex in stats["short_examples"]:
            print(f"  \"{ex}\"")

    print("\n" + "="*60)

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    if stats["n_duplicates"] > stats["total"] * 0.05:
        print(f"  ⚠️  {stats['n_duplicates']} duplicates found — run with --dedup to remove them")
    else:
        print(f"  ✅ Duplicate rate is healthy")

    if stats["n_short_responses"] > stats["total"] * 0.1:
        print(f"  ⚠️  {stats['n_short_responses']} short responses — consider raising --min-length in prepare_data.py")
    else:
        print(f"  ✅ Short response rate is acceptable")

    short_pct = stats["length_distribution"]["short (1-2 words)"] / stats["total"] * 100
    if short_pct > 30:
        print(f"  ⚠️  {short_pct:.0f}% of responses are 1-2 words — model may learn to give very short replies")
    else:
        print(f"  ✅ Response length distribution looks reasonable")

    top_count = stats["top_responses"][0][1] if stats["top_responses"] else 0
    if top_count > 50:
        print(f"  ⚠️  Most common response appears {top_count}x — could cause repetitive outputs")
    else:
        print(f"  ✅ No single response dominates the dataset")

    print()


def deduplicate(records: list[dict]) -> list[dict]:
    seen = set()
    clean = []
    for r in records:
        key = get_assistant_text(r) + "|" + get_user_text(r)
        if key not in seen:
            seen.add(key)
            clean.append(r)
    return clean


def main():
    parser = argparse.ArgumentParser(description="Analyze JSONL training dataset quality")
    parser.add_argument("--input", required=True, help="Input JSONL file to analyze")
    parser.add_argument("--dedup", action="store_true", help="Remove duplicate pairs")
    parser.add_argument("--output", default=None, help="Output path for cleaned dataset (requires --dedup)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    records = load_jsonl(str(input_path))
    print(f"Loaded {len(records):,} records")

    stats = analyze(records)
    print_report(stats)

    if args.dedup:
        clean = deduplicate(records)
        removed = len(records) - len(clean)
        print(f"Deduplication: {len(records):,} → {len(clean):,} (removed {removed:,})")

        out_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_clean.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
        print(f"Saved clean dataset to: {out_path}")


if __name__ == "__main__":
    main()
