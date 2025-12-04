import asyncio
import json
from typing import Literal, Dict, Any
from openai import OpenAI

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage  # ВАЖНО
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from config import get_settings
from prompts import TASK_ASSISTANT_SYSTEM_PROMPT
from openai import OpenAI

settings = get_settings()
bot = Bot(token=settings.bot_token)

client = OpenAI(api_key=settings.openai_api_key)

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
# ----------------- FSM-состояния ----------------- #

class AddTaskState(StatesGroup):
    waiting_voice_or_text = State()


class ReportState(StatesGroup):
    waiting_voice_or_text = State()


# ----------------- Клавиатура ----------------- #

def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="➕ Добавить задачу на день",
                    callback_data="add:day",
                )
            ],
            [
                InlineKeyboardButton(
                    text="➕ Добавить задачу на неделю",
                    callback_data="add:week",
                )
            ],
            [
                InlineKeyboardButton(
                    text="➕ Добавить задачу на месяц",
                    callback_data="add:month",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📊 Отчёт по задачам",
                    callback_data="report:auto",
                )
            ],
        ]
    )


# ----------------- Вызов OpenAI ----------------- #

async def call_task_model(
    *,
    button: Literal["add", "report"],
    period: Literal["day", "week", "month", "auto"],
    text: str,
) -> Dict[str, Any]:
    """
    Вызывает OpenAI Chat Completion с нашим системным промтом.
    Модель обязана вернуть один JSON-объект (мы его парсим).
    """

    user_payload = {
        "button": button,
        "period": period,
        "text": text,
    }

    messages = [
        {
            "role": "system",
            "content": TASK_ASSISTANT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]

    url = "https://api.openai.com/v1/chat/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_model,
                "messages": messages,
                # просим строго JSON-объект
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]

    # content — строка с JSON, парсим
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # На всякий случай: если модель нарушила договор,
        # вернем "заглушку" с ошибкой.
        raise RuntimeError(f"Model returned non-JSON content: {content}")

    return parsed


# ----------------- Вспомогательная функция STT (заглушка) ----------------- #
import tempfile
from pathlib import Path


async def transcribe_voice(message: Message) -> str:
    """
    Скачиваем voice из Telegram, отправляем в OpenAI Whisper
    и возвращаем текстовую расшифровку.
    """
    # Временный файл для voice
    tmp_dir = tempfile.gettempdir()
    tmp_path = Path(tmp_dir) / f"voice_{message.chat.id}_{message.message_id}.oga"

    # Скачиваем файл с серверов Telegram
    await bot.download(message.voice, destination=tmp_path)

    try:
        with tmp_path.open("rb") as audio_file:
            # Модель можно оставить whisper-1 или сменить на gpt-4o-mini-transcribe,
            # если она у тебя доступна.
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",  # вернёт просто строку
                # language="ru",  # можно указать язык явно, но не обязательно
            )
    finally:
        # Чистим временный файл
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Если response_format="text" — transcription уже строка
    if isinstance(transcription, str):
        return transcription

    # На случай если ответ объект с полем text
    return getattr(transcription, "text", "")


# ----------------- Хендлеры ----------------- #

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Это бот для управления задачами по голосу.\n"
        "Выбери нужное действие на инлайн-клавиатуре ниже:",
        reply_markup=main_menu_keyboard(),
    )


@dp.callback_query(F.data.startswith("add:"))
async def callback_add_task(callback: CallbackQuery, state: FSMContext):
    _, period = callback.data.split(":", maxsplit=1)
    await state.set_state(AddTaskState.waiting_voice_or_text)
    await state.update_data(period=period)

    await callback.message.answer(
        f"Отправь одно голосовое сообщение или текст с задачами на {period}.\n"
        f"Говори естественно, я сам выделю задачи.",
        reply_markup=main_menu_keyboard(),
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("report:"))
async def callback_report(callback: CallbackQuery, state: FSMContext):
    _, period = callback.data.split(":", maxsplit=1)
    await state.set_state(ReportState.waiting_voice_or_text)
    await state.update_data(period=period)

    await callback.message.answer(
        "Скажи голосом или напиши текстом, какой отчёт тебе нужен.\n"
        "Например: «покажи, что я сделал за неделю» или "
        "«какие задачи у меня ещё не выполнены на месяц».",
        reply_markup=main_menu_keyboard(),
    )
    await callback.answer()


# -------- Добавление задач: принимаем голос или текст -------- #

@dp.message(AddTaskState.waiting_voice_or_text, F.voice)
async def handle_add_tasks_voice(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data.get("period", "day")

    await message.answer("Обрабатываю голосовое сообщение, подожди немного...")

    try:
        transcript = await transcribe_voice_stub(message)
    except Exception as e:
        await message.answer(
            "Не удалось расшифровать голос. "
            "Пожалуйста, отправь текстом, что нужно сделать."
        )
        return

    await _process_add_tasks(message, period=period, text=transcript)
    await state.clear()


@dp.message(AddTaskState.waiting_voice_or_text, F.text)
async def handle_add_tasks_text(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data.get("period", "day")

    transcript = message.text
    await _process_add_tasks(message, period=period, text=transcript)
    await state.clear()


async def _process_add_tasks(message: Message, period: str, text: str):
    try:
        result = await call_task_model(
            button="add",
            period=period,  # day | week | month
            text=text,
        )
    except Exception as e:
        await message.answer(f"Ошибка при обращении к ИИ: {e}")
        return

    # На этом этапе result — это уже JSON вида:
    # {
    #   "mode": "add",
    #   "period": "...",
    #   "tasks": [...]
    # }
    # Здесь ты можешь сохранить задачи в БД.
    # Пока просто показываем JSON пользователю.
    pretty = json.dumps(result, ensure_ascii=False, indent=2)
    await message.answer(
        "Я выделил такие задачи (JSON):\n"
        f"<pre>{pretty}</pre>",
        parse_mode="HTML",
    )


# -------- Отчёт по задачам: принимаем голос или текст -------- #

@dp.message(ReportState.waiting_voice_or_text, F.voice)
async def handle_report_voice(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data.get("period", "auto")

    await message.answer("Обрабатываю голосовое сообщение, подожди немного...")

    try:
        transcript = await transcribe_voice_stub(message)
    except Exception as e:
        await message.answer(
            "Не удалось расшифровать голос. "
            "Пожалуйста, отправь текстом, какой отчёт нужен."
        )
        return

    await _process_report(message, period=period, text=transcript)
    await state.clear()


@dp.message(ReportState.waiting_voice_or_text, F.text)
async def handle_report_text(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data.get("period", "auto")

    transcript = message.text
    await _process_report(message, period=period, text=transcript)
    await state.clear()


async def _process_report(message: Message, period: str, text: str):
    try:
        result = await call_task_model(
            button="report",
            period=period,  # day | week | month | auto
            text=text,
        )
    except Exception as e:
        await message.answer(f"Ошибка при обращении к ИИ: {e}")
        return

    # result:
    # {
    #   "mode": "report",
    #   "period": "...",
    #   "status_filter": "done | not_done | all"
    # }
    # Здесь ты можешь дернуть свою БД и реально отдать отчёт.
    # Пока просто возвращаем JSON для отладки.
    pretty = json.dumps(result, ensure_ascii=False, indent=2)
    await message.answer(
        "Параметры отчёта (JSON):\n"
        f"<pre>{pretty}</pre>",
        parse_mode="HTML",
    )


# ----------------- Точка входа ----------------- #

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
