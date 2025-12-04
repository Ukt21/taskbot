import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    bot_token: str
    openai_api_key: str
    openai_model: str


def get_settings() -> Settings:
    bot_token = os.getenv("BOT_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # можешь заменить на свою модель

    if not bot_token:
        raise RuntimeError("BOT_TOKEN is not set in environment")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")

    return Settings(
        bot_token=bot_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )
