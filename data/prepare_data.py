"""
prepare_data.py — Parse Facebook Messenger JSON into fine-tuning JSONL.

Expected JSON format per file:
{
  "messages": [
    {
      "senderName": "Your Name",
      "text": "hbd",
      "timestamp": 1711599369687,
      "type": "text",
      "isUnsent": false,
      "media": [],
      "reactions": []
    },
    ...
  ]
}

Messages are in chronological order (ascending timestamps).

USAGE:
    python data/prepare_data.py \
        --inbox "C:/path/to/inbox" \
        --your-name "Your Name" \
        --output data/train.jsonl
"""

import json
import argparse
import random
import re
import glob
from pathlib import Path


def load_conversation(json_path: str) -> list[dict]:
    """Load messages from a single JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def build_chat_pairs(
    messages: list[dict],
    your_name: str,
    min_length: int = 2,
) -> list[dict]:
    """
    Convert a thread into (prompt, response) pairs where YOU are the assistant.

    Improvements:
    - Filters out pairs where prompt or response is too short
    - Filters out responses that look like load requests, phone numbers, or noise
    - Skips pairs where messages are too far apart in time (different topics)
    - Concatenates consecutive messages from the same sender before pairing
    """
    # Filter to text-only, non-unsent messages with actual content
    messages = [
        m for m in messages
        if m.get("type") == "text"
        and not m.get("isUnsent", False)
        and m.get("text", "").strip()
    ]

    # Sort by timestamp ascending (chronological)
    messages.sort(key=lambda m: m.get("timestamp", 0))

    # Merge consecutive messages from the same sender into one
    # (people often send multiple short messages in a row)
    merged = []
    for msg in messages:
        if merged and merged[-1].get("senderName") == msg.get("senderName"):
            # Check time gap — only merge if within 3 minutes
            time_gap = msg.get("timestamp", 0) - merged[-1].get("timestamp", 0)
            if time_gap < 3 * 60 * 1000:  # 3 minutes in ms
                merged[-1]["text"] = merged[-1]["text"] + " " + msg["text"]
                merged[-1]["timestamp"] = msg["timestamp"]  # update to latest
                continue
        merged.append(dict(msg))
    messages = merged

    pairs = []
    i = 0
    while i < len(messages) - 1:
        msg = messages[i]
        sender = msg.get("senderName", "")
        text = msg.get("text", "").strip()

        # We want: someone else says something → you reply
        if sender != your_name and len(text) >= min_length:
            # Look ahead for your first response
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                next_sender = next_msg.get("senderName", "")
                next_text = next_msg.get("text", "").strip()

                # Skip if too much time passed (>30 min = different conversation topic)
                time_gap = next_msg.get("timestamp", 0) - msg.get("timestamp", 0)
                if time_gap > 30 * 60 * 1000:
                    break

                if next_sender == your_name and len(next_text) >= min_length:
                    # Quality filters — skip low-quality pairs
                    if _is_quality_pair(text, next_text):
                        pairs.append({
                            "prompt": text,
                            "response": next_text,
                        })
                    i = j
                    break
                elif next_sender != your_name:
                    break
                j += 1
        i += 1

    return pairs


# Patterns to filter out noisy responses
_NOISE_PATTERNS = [
    r"^\d{10,}$",                    # pure phone numbers
    r"^(load|paloadi|pa-load)",      # load requests
    r"^(ok|oo|haha|hehe|lol|wow)$",  # single filler words (too short to learn from)
    r"^\W+$",                         # only punctuation/symbols
]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]

def _is_quality_pair(prompt: str, response: str) -> bool:
    """Return True if this prompt/response pair is worth training on."""
    # Both must be at least 3 characters
    if len(prompt.strip()) < 3 or len(response.strip()) < 3:
        return False

    # Response must be at least 2 words OR meaningful short reply
    response_words = response.strip().split()
    if len(response_words) < 2 and len(response.strip()) < 8:
        return False

    # Filter noise patterns
    for pattern in _NOISE_RE:
        if pattern.search(response.strip()):
            return False

    # Prompt shouldn't be just a single character/emoji
    if len(prompt.strip()) < 3:
        return False

    return True


def format_as_chatml(prompt: str, response: str) -> dict:
    """Format as ChatML — compatible with Qwen2.5, LLaMA-3, Phi-3, Mistral."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are replying to a message in a casual chat conversation. "
                    "Match the tone, style, and language of the user's typical responses. "
                    "Be natural, concise, and conversational."
                ),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Facebook messages for fine-tuning")
    parser.add_argument(
        "--inbox",
        required=True,
        help="Path to the inbox folder containing per-person JSON files",
    )
    parser.add_argument(
        "--your-name",
        required=True,
        help="Your name exactly as it appears in senderName (case-sensitive)",
    )
    parser.add_argument(
        "--output",
        default="data/train.jsonl",
        help="Output JSONL base path (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum character length for a message (default: 2)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap total training pairs (e.g. 10000 for faster training)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of data for validation (default: 0.05)",
    )
    args = parser.parse_args()

    inbox_path = Path(args.inbox)
    if not inbox_path.exists():
        print(f"ERROR: Inbox path not found: {inbox_path}")
        return

    # Find all JSON files — could be flat or nested in subfolders
    json_files = list(inbox_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON file(s) in {inbox_path}\n")

    if not json_files:
        print("No JSON files found. Check your --inbox path.")
        return

    all_pairs = []
    for json_file in sorted(json_files):
        try:
            messages = load_conversation(str(json_file))
            if not messages:
                continue
            pairs = build_chat_pairs(messages, args.your_name, args.min_length)
            if pairs:
                rel = json_file.relative_to(inbox_path)
                print(f"  {rel}: {len(pairs)} pairs")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  SKIP {json_file.name}: {e}")

    print(f"\nTotal pairs before cap: {len(all_pairs)}")

    if len(all_pairs) == 0:
        print("\nNo pairs found. Check that --your-name matches exactly.")
        print("Tip: open any JSON file and copy the exact senderName value for yourself.")
        return

    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # Cap
    if args.max_pairs:
        all_pairs = all_pairs[: args.max_pairs]
        print(f"Capped to: {len(all_pairs)} pairs")

    # Train / val split
    split_idx = max(1, int(len(all_pairs) * (1 - args.val_split)))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_path = output_path.with_name(output_path.stem + "_train.jsonl")
    val_path = output_path.with_name(output_path.stem + "_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(format_as_chatml(pair["prompt"], pair["response"]), ensure_ascii=False))
            f.write("\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(format_as_chatml(pair["prompt"], pair["response"]), ensure_ascii=False))
            f.write("\n")

    print(f"\nSaved:")
    print(f"  Train : {train_path} ({len(train_pairs)} examples)")
    print(f"  Val   : {val_path}  ({len(val_pairs)} examples)")
    print(f"\nNext: upload both JSONL files to RunPod and run training/finetune.py")


if __name__ == "__main__":
    main()
