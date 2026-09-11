"""
finetune.py — QLoRA fine-tune on RunPod using your Facebook message data.

BASE MODEL: unsloth/Phi-3.5-mini-instruct  (3.8B, fast, good for chat style)
             Change BASE_MODEL below to swap to LLaMA-3.1-8B or Mistral-7B.

USAGE on RunPod:
    python training/finetune.py \
        --train data/train_train.jsonl \
        --val   data/train_val.jsonl \
        --output ./output

All hyperparams can be overridden via CLI flags (see --help).
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass

# ─── Lazy imports (installed on RunPod only) ────────────────────────────────
try:
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run:  bash training/runpod_setup.sh  first.")
    raise


# ─── Defaults ────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"     # swap to "unsloth/Phi-3.5-mini-instruct" or "unsloth/Meta-Llama-3.1-8B-Instruct" etc.
MAX_SEQ_LENGTH = 1024                           # keep low for chat — saves VRAM
LORA_RANK = 32                                  # 16 or 32 works well for style adaptation
LORA_ALPHA = 64


@dataclass
class TrainConfig:
    base_model: str
    train_file: str
    val_file: str
    output_dir: str
    num_epochs: int
    batch_size: int
    grad_accumulation: int
    learning_rate: float
    warmup_steps: int
    save_steps: int
    max_seq_length: int
    lora_rank: int
    lora_alpha: int
    load_in_4bit: bool
    push_to_hub: bool
    hub_model_id: str


def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file where each line has a `messages` key (ChatML format)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    print(f"Loaded {len(records)} examples from {path}")
    return Dataset.from_list(records)


def format_chatml_prompt(example, tokenizer):
    """Apply the model's chat template to the messages list."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(config: TrainConfig):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {config.base_model}")
    print(f"  Output:      {config.output_dir}")
    print(f"  Epochs:      {config.num_epochs}")
    print(f"  LoRA rank:   {config.lora_rank}")
    print(f"{'='*60}\n")

    # ── 1. Load base model with Unsloth (4-bit QLoRA) ─────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,                 # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=config.load_in_4bit,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,          # 0 is optimal with Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    print(model.print_trainable_parameters())

    # ── 2. Load and format datasets ────────────────────────────────────────
    train_ds = load_jsonl(config.train_file)
    val_ds = load_jsonl(config.val_file)

    # Apply chat template
    train_ds = train_ds.map(lambda ex: format_chatml_prompt(ex, tokenizer), batched=False)
    val_ds = val_ds.map(lambda ex: format_chatml_prompt(ex, tokenizer), batched=False)

    # ── 3. Training arguments ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,              # use bf16 on A100/H100; set fp16=True for older GPUs
        fp16=False,
        optim="adamw_8bit",     # memory-efficient optimizer (Unsloth)
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",       # change to "wandb" if you want W&B logging
    )

    # ── 4. SFT Trainer ────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,              # packing can help with speed but tricky for chat
        args=training_args,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    trainer_stats = trainer.train()
    print(f"\nTraining complete. Stats: {trainer_stats}")

    # ── 6. Save ──────────────────────────────────────────────────────────
    output_path = Path(config.output_dir)
    adapter_path = output_path / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved to: {adapter_path}")

    # Save merged model (optional — larger but simpler to load)
    merged_path = output_path / "merged_model"
    print(f"Merging and saving full model to: {merged_path}")
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",     # merged_4bit_forced for smaller size
    )
    print("Merged model saved.")

    # ── 7. Optional push to HuggingFace Hub ──────────────────────────────
    if config.push_to_hub and config.hub_model_id:
        print(f"\nPushing to Hub: {config.hub_model_id}")
        model.push_to_hub_merged(
            config.hub_model_id,
            tokenizer,
            save_method="lora",
            token=os.environ.get("HF_TOKEN", ""),
        )
        print("Pushed to Hub.")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune on RunPod")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--train", required=True, help="Path to train JSONL")
    parser.add_argument("--val", required=True, help="Path to val JSONL")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="", help="e.g. your-hf-username/my-chat-model")
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.base_model,
        train_file=args.train,
        val_file=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        load_in_4bit=not args.no_4bit,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    train(config)


if __name__ == "__main__":
    main()
