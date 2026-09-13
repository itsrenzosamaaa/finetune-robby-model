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

    Finds messages from others → looks for your next reply.
    """
    # Filter to text-only, non-unsent messages with actual content
    messages = [
        m for m in messages
        if m.get("type") == "text"
        and not m.get("isUnsent", False)
        and m.get("text", "").strip()
    ]

    # Sort by timestamp just in case (ascending = chronological)
    messages.sort(key=lambda m: m.get("timestamp", 0))

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

                if next_sender == your_name and len(next_text) >= min_length:
                    pairs.append({
                        "prompt": text,
                        "response": next_text,
                    })
                    i = j  # advance past the consumed response
                    break
                elif next_sender != your_name:
                    # Someone else replied before you — skip this exchange
                    break
                j += 1
        i += 1

    return pairs


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
