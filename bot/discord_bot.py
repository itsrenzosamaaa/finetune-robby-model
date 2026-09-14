"""
discord_bot.py — Discord bot that uses your fine-tuned model locally.

The bot loads your merged model (or LoRA adapter) and responds to:
  - Direct messages
  - Messages in channels where it's mentioned (@BotName)
  - A !chat <message> command

SETUP:
    1. Copy .env.example to .env and fill in your values
    2. pip install -r bot/requirements.txt
    3. python bot/discord_bot.py

GETTING A DISCORD BOT TOKEN:
    1. https://discord.com/developers/applications → New Application
    2. Bot tab → Add Bot → Copy token → paste in .env
    3. OAuth2 → URL Generator → bot + Send Messages + Read Message History
    4. Open the generated URL to invite the bot to your server
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
MODEL_PATH = os.getenv("MODEL_PATH", "./output/merged_model")
USE_LORA_ONLY = os.getenv("USE_LORA_ONLY", "false").lower() == "true"
BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!")

# Only respond to these channel IDs (comma-separated). Leave empty = all channels.
ALLOWED_CHANNEL_IDS_RAW = os.getenv("ALLOWED_CHANNEL_IDS", "")
ALLOWED_CHANNEL_IDS = (
    set(int(x.strip()) for x in ALLOWED_CHANNEL_IDS_RAW.split(",") if x.strip())
    if ALLOWED_CHANNEL_IDS_RAW
    else set()
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─── Model loading ────────────────────────────────────────────────────────────
class ChatModel:
    """Wraps the fine-tuned model for inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        log.info(f"Loading model from: {MODEL_PATH}")
        try:
            # Try Unsloth first (faster inference)
            from unsloth import FastLanguageModel

            if USE_LORA_ONLY:
                log.info("Loading base model + LoRA adapter via Unsloth...")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=BASE_MODEL,
                    max_seq_length=1024,
                    dtype=None,
                    load_in_4bit=True,
                )
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, MODEL_PATH)
            else:
                log.info("Loading merged model via Unsloth...")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=MODEL_PATH,
                    max_seq_length=1024,
                    dtype=None,
                    load_in_4bit=False,     # merged models are usually float16
                )

            FastLanguageModel.for_inference(self.model)

        except ImportError:
            # Fallback: plain HuggingFace transformers (slower, no Unsloth)
            log.warning("Unsloth not found — falling back to vanilla transformers (slower)")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            from pathlib import Path

            model_path = Path(MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at: {MODEL_PATH}\n"
                    f"Run: huggingface-cli download itsrenzosamaaa/robby-chat-model --local-dir output/merged_model"
                )

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self._loaded = True
        log.info("Model loaded successfully.")

    def generate(self, user_message: str) -> str:
        """Run inference on a single user message."""
        if not self._loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model is not loaded yet — try again in a moment")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are replying to a message in a casual chat conversation. "
                    "Match the tone, style, and language of the user's typical responses. "
                    "Be natural, concise, and conversational."
                ),
            },
            {"role": "user", "content": user_message},
        ]

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        import torch

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()


# ─── Discord Bot ──────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)
chat_model = ChatModel()


def is_allowed_channel(channel_id: int) -> bool:
    """Return True if the bot should respond in this channel."""
    if not ALLOWED_CHANNEL_IDS:
        return True  # no restriction — respond everywhere
    return channel_id in ALLOWED_CHANNEL_IDS


@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    log.info("Loading model in background...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, chat_model.load)
    log.info("Bot is ready!")


@bot.event
async def on_message(message: discord.Message):
    # Never respond to ourselves
    if message.author == bot.user:
        return

    # Process commands first (e.g. !chat)
    await bot.process_commands(message)
    # Respond to DMs, @mentions, or any message in allowed channels
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user in message.mentions
    is_allowed = is_allowed_channel(message.channel.id)

    if not (is_dm or is_mention or (is_allowed and ALLOWED_CHANNEL_IDS)):
        return

    # Strip the @mention from the message
    content = message.content
    if is_mention:
        content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()

    if not content:
        await message.reply("ano ba yan, wala kang sinabi 🙄")
        return

    # Model not ready yet — tell the user instead of crashing
    if not chat_model._loaded:
        await message.reply("sandali lang, naglo-load pa ako 😴 try mo ulit after 1 minute")
        return

    # Generate response — run in executor to avoid blocking the event loop
    await message.add_reaction("🤔")  # thinking reaction while generating
    async with message.channel.typing():
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, chat_model.generate, content)
            await message.remove_reaction("🤔", bot.user)  # remove thinking emoji
        except Exception as e:
            log.error(f"Generation error: {e}", exc_info=True)
            await message.remove_reaction("🤔", bot.user)
            await message.add_reaction("💀")
            response = "nagcrash ako HAHAHA antay ka muna bro 💀"

    await message.reply(response)


@bot.command(name="chat")
async def chat_command(ctx: commands.Context, *, message: str):
    """!chat <message> — Chat with the fine-tuned model."""
    if not is_allowed_channel(ctx.channel.id):
        return

    if not chat_model._loaded:
        await ctx.reply("sandali lang, naglo-load pa ako 😴 try mo ulit after 1 minute")
        return

    async with ctx.typing():
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, chat_model.generate, message)
        except Exception as e:
            log.error(f"Generation error: {e}", exc_info=True)
            response = "uy nag-error yung utak ko, try mo ulit 😭"

    await ctx.reply(response)


@bot.command(name="ping")
async def ping_command(ctx: commands.Context):
    """!ping — Check if the bot is alive."""
    await ctx.reply(f"Pong! Latency: {round(bot.latency * 1000)}ms")


@bot.command(name="model")
async def model_info_command(ctx: commands.Context):
    """!model — Show which model is loaded."""
    status = "loaded" if chat_model._loaded else "not loaded yet"
    await ctx.reply(f"Model: `{MODEL_PATH}` ({status})")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("ERROR: DISCORD_TOKEN not set in .env")
        sys.exit(1)

    log.info("Starting Discord bot...")
    bot.run(DISCORD_TOKEN)
