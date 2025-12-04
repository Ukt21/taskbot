import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import httpx
from openai import OpenAI

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import get_settings
from prompts import TASK_ASSISTANT_SYSTEM_PROMPT
from db import init_db, add_task, get_tasks, set_task_done, delete_task


# ---------------- ИНИЦИАЛИЗАЦИЯ ----------------

settings = get_settings()
bot = Bot(token=settings.bot_token)
client = OpenAI(api_key=settings.openai_api_key)

storage = MemoryStorage()
dp = Dispatcher(storage=storage)

PERIOD_LABELS_RU = {
    "day": "день",
    "week": "неделю",
    "month": "месяц",
    "auto": "период",
}


# ---------------- КЛАВИАТУРА ----------------

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


def build_task_buttons(user_id: int, period: str | None):
    """
    Строим клавиатуру задач для пользователя и периода.
    Если period = None, берём все активные задачи.
    """
    tasks = get_tasks(user_id, period=period, only_active=True)

    if not tasks:
        return None

    kb = InlineKeyboardBuilder()
    for t in tasks:
        kb.button(text=f"✅ {t['title']}", callback_data=f"done:{t['id']}")
        kb.adjust(1)

    return kb.as_markup()


# ---------------- РАСШИФРОВКА ГОЛОСА (WHISPER) ----------------

async def transcribe_voice(message: Message) -> str:
    """
    Скачиваем voice из Telegram и отправляем в OpenAI Whisper (whisper-1).
    Возвращаем чистый текст.
    """
    tmp_path = Path(tempfile.gettempdir()) / f"voice_{message.message_id}.oga"

    # скачиваем файл с серверов Telegram
    tg_file = await bot.get_file(message.voice.file_id)
    await bot.download_file(tg_file.file_path, tmp_path)

    try:
        with tmp_path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text",  # вернёт просто строку
                # language="ru",  # можно раскомментировать, чтобы фиксировать язык
            )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result  # response_format="text" → result уже строка


# ---------------- ВЫЗОВ ИИ ДЛЯ РАЗБОРА ЗАДАЧ ----------------

async def call_task_model(button: str, period: str, text: str) -> Dict[str, Any]:
    """
    Вызывает модель чата по твоему системному промту.
    Возвращает JSON (dict) с полями mode/add/report.
    """
    payload = {
        "button": button,
        "period": period,
        "text": text,
    }

    messages = [
        {"role": "system", "content": TASK_ASSISTANT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    async with httpx.AsyncClient(timeout=60.0) as client_http:
        r = await client_http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_model,  # напр. gpt-4.1-mini
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


# ---------------- СОСТОЯНИЯ FSM ----------------

class AddTaskState(StatesGroup):
    waiting_voice_or_text = State()


class ReportState(StatesGroup):
    waiting_voice_or_text = State()


# ---------------- ХЕНДЛЕРЫ /start И МЕНЮ ----------------

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот для управления задачами по голосу.\n"
        "Выбери нужное действие:",
        reply_markup=main_menu_keyboard(),
    )


@dp.callback_query(F.data.startswith("add:"))
async def callback_add(callback: CallbackQuery, state: FSMContext):
    """
    Нажатие на «Добавить задачу на день/неделю/месяц».
    """
    _, period = callback.data.split(":", maxsplit=1)
    ru = PERIOD_LABELS_RU.get(period, period)

    await state.set_state(AddTaskState.waiting_voice_or_text)
    await state.update_data(period=period)

    await callback.message.answer(
        f"Отправь голосовое или текст с задачами на {ru}.\n"
        f"Говори естественно, я сам выделю задачи.",
        reply_markup=main_menu_keyboard(),
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("report:"))
async def callback_report(callback: CallbackQuery, state: FSMContext):
    """
    Нажатие на «Отчёт по задачам».
    """
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


# ---------------- ДОБАВЛЕНИЕ ЗАДАЧ ----------------

@dp.message(AddTaskState.waiting_voice_or_text, F.voice)
async def add_task_voice(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data["period"]

    await message.answer("Обрабатываю голосовое, подожди немного...")

    try:
        text = await transcribe_voice(message)
    except Exception as e:
        await message.answer(f"Ошибка STT: {e}")
        return

    await _process_add_tasks(message, period, text)
    await state.clear()


@dp.message(AddTaskState.waiting_voice_or_text, F.text)
async def add_task_text(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data["period"]

    await _process_add_tasks(message, period, message.text)
    await state.clear()


async def _process_add_tasks(message: Message, period: str, text: str):
    """
    Вызываем ИИ, сохраняем задачи в SQLite и показываем инлайн-кнопки.
    """
    try:
        result = await call_task_model("add", period, text)
    except Exception as e:
        await message.answer(f"Ошибка ИИ: {e}")
        return

    user_id = message.from_user.id
    tasks_list = result.get("tasks", [])

    for item in tasks_list:
        title = item.get("title", "").strip()
        if not title:
            continue
        add_task(user_id, title, period)

    kb = build_task_buttons(user_id, period)
    if kb:
        await message.answer(
            f"Задач добавлено: {len(tasks_list)}",
            reply_markup=kb,
        )
    else:
        await message.answer("Не удалось выделить задачи из текста.")


# ---------------- ОТЧЁТ ПО ЗАДАЧАМ ----------------

@dp.message(ReportState.waiting_voice_or_text, F.voice)
async def report_voice(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data["period"]

    await message.answer("Обрабатываю голосовое, подожди немного...")

    try:
        text = await transcribe_voice(message)
    except Exception as e:
        await message.answer(f"Ошибка STT: {e}")
        return

    await _process_report(message, period, text)
    await state.clear()


@dp.message(ReportState.waiting_voice_or_text, F.text)
async def report_text(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data["period"]

    await _process_report(message, period, message.text)
    await state.clear()


async def _process_report(message: Message, period: str, text: str):
    """
    Вызываем ИИ (если нужно), но пока для простоты используем period из FSM.
    Показываем активные задачи пользователя.
    """
    user_id = message.from_user.id
    # period == "auto" → показываем все периоды
    tasks = get_tasks(user_id, None if period == "auto" else period, only_active=True)

    if not tasks:
        await message.answer("У тебя пока нет активных задач.")
        return

    # Берём период первой задачи (если в отчёте auto)
    real_period = tasks[0]["period"] if period == "auto" else period
    kb = build_task_buttons(user_id, real_period)

    await message.answer("Текущие задачи:", reply_markup=kb)


# ---------------- ОБРАБОТКА КНОПОК ✅ / ❌ ----------------

@dp.callback_query(F.data.startswith("done:"))
async def cb_done(callback: CallbackQuery):
    task_id = int(callback.data.split(":", maxsplit=1)[1])
    set_task_done(task_id)
    await callback.answer("Задача отмечена как выполненная ✅")
    await refresh_after_change(callback)


@dp.callback_query(F.data.startswith("delete:"))
async def cb_delete(callback: CallbackQuery):
    task_id = int(callback.data.split(":", maxsplit=1)[1])
    delete_task(task_id)
    await callback.answer("Задача удалена ❌")
    await refresh_after_change(callback)


async def refresh_after_change(callback: CallbackQuery):
    """
    После изменения задач перестраиваем клавиатуру.
    Если задач не осталось — редактируем текст.
    """
    user_id = callback.from_user.id
    tasks = get_tasks(user_id, None, only_active=True)

    if not tasks:
        await callback.message.edit_text("🎉 Все задачи выполнены!", reply_markup=None)
        return

    # Берём период первой задачи
    period = tasks[0]["period"]
    kb = build_task_buttons(user_id, period)
    await callback.message.edit_reply_markup(reply_markup=kb)


# ---------------- MAIN ----------------

async def main():
    init_db()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
