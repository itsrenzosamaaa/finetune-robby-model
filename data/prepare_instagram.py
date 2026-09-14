"""
prepare_instagram.py — Parse Instagram DM export into fine-tuning JSONL.

Format is almost identical to Facebook Messenger export:
{
  "participants": [{"name": "Someone"}, {"name": "Ren"}],
  "messages": [
    {
      "sender_name": "Ren",
      "timestamp_ms": 1711955241101,
      "content": "ya na",
      ...
    }
  ]
}

Messages are in reverse-chronological order — script flips them.

USAGE:
    python data/prepare_instagram.py \
        --inbox "C:/path/to/your_instagram_activity/messages/inbox" \
        --your-name "Ren" \
        --output data/instagram.jsonl
"""

import json
import argparse
import random
import re
import glob
from pathlib import Path


# ── Noise patterns to skip ────────────────────────────────────────────────────
_SYSTEM_CONTENT = [
    r"^liked a message$",
    r"^reacted .+ to your message",
    r".+shared a story\.$",
    r".+sent an attachment\.$",
    r"^you shared a story\.$",
    r"^missed video chat$",
    r"^missed audio chat$",
    r"^said .+ to your story$",
]
_SYSTEM_RE = [re.compile(p, re.IGNORECASE) for p in _SYSTEM_CONTENT]

_NOISE_RESPONSE_PATTERNS = [
    r"^\d{10,}$",
    r"^(ok|oo|haha|hehe|lol|wow|ah|oh)$",
    r"^\W+$",
    r"^https?://\S+$",
]
_NOISE_RESPONSE_RE = [re.compile(p, re.IGNORECASE) for p in _NOISE_RESPONSE_PATTERNS]


def fix_encoding(text: str) -> str:
    """Fix Instagram's mangled UTF-8 (encoded as latin-1 bytes)."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def is_system_message(content: str) -> bool:
    for pattern in _SYSTEM_RE:
        if pattern.search(content.strip()):
            return True
    return False


def is_quality_response(text: str) -> bool:
    if len(text.strip()) < 3:
        return False
    words = text.strip().split()
    if len(words) < 2 and len(text.strip()) < 8:
        return False
    for pattern in _NOISE_RESPONSE_RE:
        if pattern.match(text.strip()):
            return False
    return True


def load_conversation(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def build_chat_pairs(
    messages: list[dict],
    your_name: str,
    min_length: int = 3,
) -> list[dict]:
    # Filter: only text messages, skip system messages, skip unsent
    clean = []
    for m in messages:
        if m.get("is_geoblocked_for_viewer"):
            continue
        content = fix_encoding(m.get("content", "").strip())
        # Skip messages with no text content
        if not content:
            continue
        # Skip system/noise messages
        if is_system_message(content):
            continue
        # Skip very short
        if len(content) < min_length:
            continue
        clean.append({
            "sender": fix_encoding(m.get("sender_name", "")),
            "text": content,
            "timestamp": m.get("timestamp_ms", 0),
        })

    # Instagram exports in reverse-chron — flip to chronological
    clean.sort(key=lambda m: m["timestamp"])

    # Merge consecutive messages from same sender within 3 minutes
    merged = []
    for msg in clean:
        if (merged
                and merged[-1]["sender"] == msg["sender"]
                and msg["timestamp"] - merged[-1]["timestamp"] < 3 * 60 * 1000):
            merged[-1]["text"] += " " + msg["text"]
            merged[-1]["timestamp"] = msg["timestamp"]
        else:
            merged.append(dict(msg))

    pairs = []
    i = 0
    while i < len(merged) - 1:
        msg = merged[i]
        if msg["sender"] != your_name and len(msg["text"]) >= min_length:
            j = i + 1
            while j < len(merged):
                next_msg = merged[j]
                time_gap = next_msg["timestamp"] - msg["timestamp"]
                # Skip if more than 30 minutes apart
                if time_gap > 30 * 60 * 1000:
                    break
                if next_msg["sender"] == your_name:
                    if is_quality_response(next_msg["text"]):
                        pairs.append({
                            "prompt": msg["text"],
                            "response": next_msg["text"],
                        })
                    i = j
                    break
                elif next_msg["sender"] != your_name:
                    break
                j += 1
        i += 1

    return pairs


def format_as_chatml(prompt: str, response: str) -> dict:
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
        ],
        "metadata": {"source": "instagram"},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Instagram DMs for fine-tuning")
    parser.add_argument("--inbox", required=True, help="Path to messages/inbox folder")
    parser.add_argument("--your-name", required=True, help="Your name as it appears in sender_name")
    parser.add_argument("--output", default="data/instagram.jsonl")
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    inbox = Path(args.inbox)
    if not inbox.exists():
        print(f"ERROR: Path not found: {inbox}")
        return

    json_files = sorted(inbox.rglob("message_*.json"))
    print(f"Found {len(json_files)} message file(s)\n")

    all_pairs = []
    for jf in json_files:
        try:
            messages = load_conversation(str(jf))
            pairs = build_chat_pairs(messages, args.your_name, args.min_length)
            if pairs:
                print(f"  {jf.parent.name[:40]}: {len(pairs)} pairs")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  SKIP {jf.name}: {e}")

    print(f"\nTotal pairs: {len(all_pairs)}")

    if not all_pairs:
        print("No pairs found. Check --your-name matches sender_name exactly.")
        return

    random.seed(42)
    random.shuffle(all_pairs)

    if args.max_pairs:
        all_pairs = all_pairs[: args.max_pairs]

    split_idx = max(1, int(len(all_pairs) * (1 - args.val_split)))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

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
    print(f"  Train: {train_path} ({len(train_pairs)} examples)")
    print(f"  Val:   {val_path}  ({len(val_pairs)} examples)")
    print(f"\nNext: merge with other datasets using merge_datasets.py")


if __name__ == "__main__":
    main()
