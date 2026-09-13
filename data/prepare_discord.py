"""
prepare_discord.py — Parse Discord message export into fine-tuning JSONL.

Supports two export formats:

FORMAT A — DiscordChatExporter (third-party tool, recommended):
    Each messages.json is a flat array:
    [
      {
        "ID": 1469588008247365705,
        "Timestamp": "2026-02-07 06:58:04",
        "Contents": "hello is there anyone...",
        "Attachments": ""
      },
      ...
    ]
    channel.json or index.json may contain channel name.
    Since ALL messages are yours, we use sliding context window:
    last N messages = prompt, your next message = response.

FORMAT B — Official Discord data request (settings → privacy → request data):
    messages/c<channel_id>/messages.csv
    All messages are also yours (your account export).

USAGE:
    python data/prepare_discord.py \
        --inbox "C:/path/to/discord-messages" \
        --output data/discord.jsonl \
        --context-window 3

The --context-window flag controls how many previous messages
are included as context before your response.
"""

import json
import csv
import argparse
import random
from pathlib import Path
from datetime import datetime


def load_discord_chatexporter(json_path: str) -> list[dict]:
    """Load DiscordChatExporter format — flat array of messages."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    messages = []
    for m in data:
        content = str(m.get("Contents", "") or "").strip()
        if not content or content == "":
            continue
        # Parse timestamp
        ts_raw = m.get("Timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw).timestamp() * 1000
        except Exception:
            ts = 0

        messages.append({
            "text": content,
            "timestamp": ts,
        })

    return sorted(messages, key=lambda x: x["timestamp"])


def load_discord_official_csv(csv_path: str) -> list[dict]:
    """Load official Discord data export — CSV format."""
    messages = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = (row.get("Contents") or row.get("content") or "").strip()
            if not content:
                continue
            ts_raw = row.get("Timestamp") or row.get("timestamp") or ""
            try:
                ts = datetime.fromisoformat(ts_raw).timestamp() * 1000
            except Exception:
                ts = 0
            messages.append({"text": content, "timestamp": ts})

    return sorted(messages, key=lambda x: x["timestamp"])


def build_context_pairs(
    messages: list[dict],
    context_window: int = 3,
    min_length: int = 5,
    max_gap_minutes: int = 60,
) -> list[dict]:
    """
    Build training pairs using a sliding context window.

    Since all messages are from you, we treat it as:
    - context = last `context_window` messages
    - response = next message

    This teaches the model to continue a conversation in your style.
    """
    # Filter short/empty messages
    messages = [m for m in messages if len(m["text"].strip()) >= min_length]

    pairs = []
    for i in range(context_window, len(messages)):
        response_msg = messages[i]
        context_msgs = messages[i - context_window: i]

        # Skip if too much time between last context message and response
        time_gap = response_msg["timestamp"] - context_msgs[-1]["timestamp"]
        if time_gap > max_gap_minutes * 60 * 1000:
            continue

        # Skip noise
        response_text = response_msg["text"].strip()
        if not _is_quality_message(response_text):
            continue

        # Build prompt from context window
        if context_window == 1:
            prompt = context_msgs[0]["text"].strip()
        else:
            prompt = "\n".join(m["text"].strip() for m in context_msgs)

        pairs.append({
            "prompt": prompt,
            "response": response_text,
        })

    return pairs


import re
_NOISE_PATTERNS = [
    r"^\d{10,}$",
    r"^(ok|oo|haha|hehe|lol|wow|ah|oh)$",
    r"^\W+$",
    r"^https?://\S+$",   # pure URLs
    r"^<@\d+>$",         # pure Discord mentions
    r"^<:\w+:\d+>$",     # pure Discord emojis
]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


def _is_quality_message(text: str) -> bool:
    if len(text.strip()) < 5:
        return False
    for pattern in _NOISE_RE:
        if pattern.match(text.strip()):
            return False
    return True


def format_as_chatml(prompt: str, response: str, source: str = "discord") -> dict:
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
        "metadata": {"source": source},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Discord messages for fine-tuning")
    parser.add_argument("--inbox", required=True, help="Path to discord-messages folder")
    parser.add_argument("--output", default="data/discord.jsonl")
    parser.add_argument("--context-window", type=int, default=2,
                        help="Number of previous messages to use as context (default: 2)")
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-gap-minutes", type=int, default=60,
                        help="Max minutes between messages to be considered same conversation")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    inbox = Path(args.inbox)
    if not inbox.exists():
        print(f"ERROR: Path not found: {inbox}")
        return

    # Find all message files
    json_files = list(inbox.rglob("messages.json"))
    csv_files = list(inbox.rglob("messages.csv"))
    print(f"Found {len(json_files)} JSON + {len(csv_files)} CSV message files\n")

    all_pairs = []

    for jf in sorted(json_files):
        try:
            msgs = load_discord_chatexporter(str(jf))
            if not msgs:
                continue
            pairs = build_context_pairs(
                msgs,
                context_window=args.context_window,
                min_length=args.min_length,
                max_gap_minutes=args.max_gap_minutes,
            )
            if pairs:
                print(f"  {jf.parent.name}: {len(pairs)} pairs")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  SKIP {jf}: {e}")

    for cf in sorted(csv_files):
        try:
            msgs = load_discord_official_csv(str(cf))
            if not msgs:
                continue
            pairs = build_context_pairs(
                msgs,
                context_window=args.context_window,
                min_length=args.min_length,
                max_gap_minutes=args.max_gap_minutes,
            )
            if pairs:
                print(f"  {cf.parent.name}: {len(pairs)} pairs")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  SKIP {cf}: {e}")

    print(f"\nTotal pairs: {len(all_pairs)}")

    if not all_pairs:
        print("No pairs found. Check your --inbox path and file format.")
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
            f.write(json.dumps(
                format_as_chatml(pair["prompt"], pair["response"], "discord"),
                ensure_ascii=False
            ))
            f.write("\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(
                format_as_chatml(pair["prompt"], pair["response"], "discord"),
                ensure_ascii=False
            ))
            f.write("\n")

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_pairs)} examples)")
    print(f"  Val:   {val_path}  ({len(val_pairs)} examples)")


if __name__ == "__main__":
    main()
