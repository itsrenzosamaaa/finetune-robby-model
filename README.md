# finetune-my-model

Fine-tune a small LLM on your Facebook messages, then run it as a Discord bot locally.

## Overview

```
Facebook export  →  prepare_data.py  →  train_train.jsonl + train_val.jsonl
                                               ↓
                                         RunPod (finetune.py)
                                               ↓
                                      output/merged_model/
                                               ↓
                                    Local Discord bot (discord_bot.py)
```

---

## Step 1 — Export Your Facebook Messages

1. Go to **Facebook → Settings → Your Facebook information → Download your information**
2. Set format to **JSON**, date range to **All time**
3. Deselect everything except **Messages**
4. Request download, wait for the email, download and extract the ZIP

Your messages will be at:
```
messages/inbox/<conversation-name>/message_1.json
```

---

## Step 2 — Prepare Training Data (run locally)

```bash
pip install tqdm   # only dep needed locally

python data/prepare_data.py \
  --inbox "C:/path/to/messages/inbox" \
  --your-name "Your Name" \
  --output data/train.jsonl
```

This creates:
- `data/train_train.jsonl` — 95% of your message pairs
- `data/train_val.jsonl` — 5% for validation

> **Privacy:** These files contain your personal messages. Keep them local. Never commit them to git (they're in `.gitignore`).

---

## Step 3 — Fine-tune on RunPod

### 3a. Create a RunPod instance
- Template: **RunPod PyTorch 2.4** (or any recent PyTorch template)
- GPU recommendation:
  | GPU | VRAM | Estimated time (3 epochs, ~1k pairs) |
  |-----|------|---------------------------------------|
  | RTX 3090 / 4090 | 24 GB | ~15–30 min |
  | A100 40GB | 40 GB | ~10–20 min |
  | RTX 4090 | 24 GB | ~15–25 min |
- Disk: at least **30 GB** (model weights)

### 3b. Upload your files to RunPod

In the RunPod web terminal or via `scp`:
```bash
# From your local machine:
scp -P <port> data/train_train.jsonl root@<pod-ip>:/workspace/data/
scp -P <port> data/train_val.jsonl   root@<pod-ip>:/workspace/data/
scp -P <port> training/finetune.py   root@<pod-ip>:/workspace/training/
scp -P <port> training/runpod_setup.sh root@<pod-ip>:/workspace/training/
```

Or use the **RunPod file browser** to upload.

### 3c. Install dependencies
```bash
bash training/runpod_setup.sh
```

### 3d. Run training
```bash
python training/finetune.py \
  --train data/train_train.jsonl \
  --val   data/train_val.jsonl \
  --output ./output \
  --epochs 3 \
  --batch-size 4
```

#### Common flags
| Flag | Default | Notes |
|------|---------|-------|
| `--base-model` | `unsloth/Phi-3.5-mini-instruct` | Swap to `unsloth/Meta-Llama-3.1-8B-Instruct` for more capability |
| `--epochs` | 3 | More epochs = more overfitting to your style |
| `--batch-size` | 4 | Lower if you get OOM errors |
| `--lora-rank` | 32 | 16 is faster, 64 captures more style nuance |
| `--push-to-hub` | false | Add `--push-to-hub --hub-model-id username/model-name` to upload to HF Hub |

### 3e. Download your model
```bash
# From RunPod to local — zip first for easier transfer:
zip -r output_model.zip output/merged_model

# Then scp it down or use RunPod's file browser
scp -P <port> root@<pod-ip>:/workspace/output_model.zip ./
```

Extract to `output/merged_model/` in this repo.

---

## Step 4 — Run the Discord Bot Locally

### 4a. Create a Discord bot
1. Go to https://discord.com/developers/applications → **New Application**
2. **Bot** tab → **Add Bot** → copy the **Token**
3. **OAuth2 → URL Generator** → check `bot` scope + permissions:
   - Send Messages
   - Read Message History
   - Read Messages/View Channels
4. Open the generated URL to invite the bot to your server

### 4b. Configure

```bash
cp bot/.env.example bot/.env
# Edit bot/.env and set:
#   DISCORD_TOKEN=your-token
#   MODEL_PATH=./output/merged_model
```

### 4c. Install dependencies

```bash
pip install -r bot/requirements.txt
```

### 4d. Run

```bash
python bot/discord_bot.py
```

### Bot commands

| Trigger | What it does |
|---------|-------------|
| `@BotName <message>` | Bot replies in the channel |
| DM the bot | Bot replies in DM |
| `!chat <message>` | Explicit chat command |
| `!ping` | Check if bot is alive |
| `!model` | Show which model is loaded |

---

## Project Structure

```
finetune-my-model/
├── data/
│   ├── prepare_data.py        # Parse Facebook JSON → JSONL
│   ├── train_train.jsonl      # Generated (gitignored)
│   └── train_val.jsonl        # Generated (gitignored)
├── training/
│   ├── finetune.py            # QLoRA training script (run on RunPod)
│   ├── requirements.txt       # RunPod Python deps
│   └── runpod_setup.sh        # One-shot RunPod setup script
├── bot/
│   ├── discord_bot.py         # Local Discord bot
│   ├── requirements.txt       # Local Python deps
│   └── .env.example           # Config template
├── output/                    # Downloaded model (gitignored)
│   └── merged_model/
└── README.md
```

---

## Model choices

| Model | Size | Style | HuggingFace ID |
|-------|------|-------|----------------|
| **Qwen 2.5 3B Instruct *(default)*** | 3B | Best for Tagalog/English mix | `unsloth/Qwen2.5-3B-Instruct` |
| Phi-3.5 Mini Instruct | 3.8B | Fast, good chat | `unsloth/Phi-3.5-mini-instruct` |
| LLaMA 3.2 3B Instruct | 3B | Solid small model | `unsloth/Llama-3.2-3B-Instruct` |
| LLaMA 3.1 8B Instruct | 8B | Best quality (RunPod only) | `unsloth/Meta-Llama-3.1-8B-Instruct` |

Change with `--base-model <HuggingFace ID>` when running `finetune.py`.

---

## Tips

- **More data = better style capture.** If you only have a few hundred message pairs, reduce `--lora-rank 16` and `--epochs 2` to avoid overfitting.
- **Multilingual?** If your Facebook messages mix languages (e.g. Tagalog + English), use `unsloth/Qwen2.5-7B-Instruct` — it handles code-switching better.
- **Temperature:** Raise `TEMPERATURE` in `.env` (0.9–1.0) for more varied/creative responses. Lower (0.6–0.7) for more consistent style.
- **ALLOWED_CHANNEL_IDS:** Set this in `.env` so the bot only responds in your test channel, not every server channel.
