"""
prepare_data.py — Parse Facebook Messenger export into fine-tuning JSONL.

HOW TO GET YOUR FACEBOOK DATA:
1. Facebook → Settings → Your Facebook information → Download your information
2. Select: Messages only, JSON format, Date range: All time
3. Download and extract the ZIP
4. Your messages are at: messages/inbox/<conversation>/message_1.json

USAGE:
    python data/prepare_data.py \
        --inbox "path/to/messages/inbox" \
        --your-name "Your Name" \
        --output "data/train.jsonl" \
        --min-length 5
"""

import json
import argparse
import os
import glob
from pathlib import Path


def load_conversation(json_path: str) -> list[dict]:
    """Load a single Facebook message JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def fix_encoding(text: str) -> str:
    """Fix Facebook's mangled UTF-8 encoding (they encode UTF-8 as latin-1 bytes)."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def build_chat_pairs(
    messages: list[dict],
    your_name: str,
    min_length: int = 5,
) -> list[dict]:
    """
    Convert a thread into (prompt, response) pairs where YOU are the assistant.

    Strategy: find a message from someone else, then find the next reply from you.
    This teaches the model to respond in your style.
    """
    # Messages come in reverse-chronological order from Facebook — flip them.
    messages = list(reversed(messages))

    pairs = []
    i = 0
    while i < len(messages) - 1:
        msg = messages[i]
        # Skip non-text messages (photos, stickers, etc.)
        if "content" not in msg:
            i += 1
            continue

        sender = fix_encoding(msg.get("sender_name", ""))
        content = fix_encoding(msg["content"])

        # We want: someone says something → you reply
        if sender != your_name and len(content) >= min_length:
            # Look ahead for your first response
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                if "content" not in next_msg:
                    j += 1
                    continue
                next_sender = fix_encoding(next_msg.get("sender_name", ""))
                next_content = fix_encoding(next_msg["content"])
                if next_sender == your_name and len(next_content) >= min_length:
                    pairs.append({
                        "prompt": content.strip(),
                        "response": next_content.strip(),
                    })
                    i = j  # advance past the response we just consumed
                    break
                elif next_sender != your_name:
                    # Someone else replied before you — skip this exchange
                    break
                j += 1
        i += 1

    return pairs


def format_as_chatml(prompt: str, response: str) -> dict:
    """
    Format as ChatML — compatible with most instruct models (Mistral, LLaMA-3, Phi-3).

    The system prompt tells the model to respond like you.
    """
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
        help="Path to the Facebook messages/inbox folder",
    )
    parser.add_argument(
        "--your-name",
        required=True,
        help="Your name exactly as it appears in Facebook (case-sensitive)",
    )
    parser.add_argument(
        "--output",
        default="data/train.jsonl",
        help="Output JSONL file path (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum character length for a message to be included (default: 5)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap total training pairs (useful for quick tests)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of data to use as validation set (default: 0.05)",
    )
    args = parser.parse_args()

    inbox_path = Path(args.inbox)
    if not inbox_path.exists():
        print(f"ERROR: Inbox path not found: {inbox_path}")
        return

    # Find all message JSON files (Facebook splits large threads into message_1.json, message_2.json, etc.)
    json_files = sorted(glob.glob(str(inbox_path / "**" / "message_*.json"), recursive=True))
    print(f"Found {len(json_files)} message file(s) in {inbox_path}")

    all_pairs = []
    for json_file in json_files:
        messages = load_conversation(json_file)
        pairs = build_chat_pairs(messages, args.your_name, args.min_length)
        all_pairs.extend(pairs)
        print(f"  {Path(json_file).parent.name}: {len(pairs)} pairs")

    print(f"\nTotal pairs: {len(all_pairs)}")

    if args.max_pairs:
        all_pairs = all_pairs[: args.max_pairs]
        print(f"Capped to: {len(all_pairs)} pairs")

    # Shuffle for good measure
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    # Train / validation split
    split_idx = max(1, int(len(all_pairs) * (1 - args.val_split)))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Write output
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
    print("\nNext step: Upload both JSONL files to RunPod before running finetune.py")


if __name__ == "__main__":
    main()
