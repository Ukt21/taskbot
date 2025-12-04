import asyncio
import logging
import os
import sqlite3
import io
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "tasks.db")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# DB INIT
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()

    # Таблица задач
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            period TEXT NOT NULL,      -- day / week / month
            is_done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # Таблица отчётов
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            ai_text TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def get_db_connection():
    return sqlite3.connect(DATABASE_PATH)


# ---------------------------------------------------------------------------
# FSM STATES
# ---------------------------------------------------------------------------

class TaskStates(StatesGroup):
    waiting_for_task_content = State()  # голос/текст новой задачи


class ReportStates(StatesGroup):
    waiting_for_report_text = State()   # черновик отчёта


# ---------------------------------------------------------------------------
# KEYBOARDS
# ---------------------------------------------------------------------------

def main_menu_keyboard() -> InlineKeyboardMarkup:
    kb = [
        [
            InlineKeyboardButton(text="➕ Задача на день", callback_data="add_task:day"),
        ],
        [
            InlineKeyboardButton(text="➕ Задача на неделю", callback_data="add_task:week"),
        ],
        [
            InlineKeyboardButton(text="➕ Задача на месяц", callback_data="add_task:month"),
        ],
        [
            InlineKeyboardButton(text="📋 Мои задачи", callback_data="show_tasks"),
        ],
        [
            InlineKeyboardButton(text="📊 Отчёт дня с ИИ", callback_data="daily_report_ai"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)


def task_inline_keyboard(task_id: int) -> InlineKeyboardMarkup:
    kb = [
        [
            InlineKeyboardButton(text="✔️ Готово", callback_data=f"task_done:{task_id}"),
            InlineKeyboardButton(text="❌ Удалить", callback_data=f"task_delete:{task_id}"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)


# ---------------------------------------------------------------------------
# OPENAI HELPERS
# ---------------------------------------------------------------------------

async def transcribe_voice(file_bytes: bytes) -> str:
    """
    Расшифровка голосового сообщения через Whisper (whisper-1).
    """
    bio = io.BytesIO(file_bytes)
    bio.name = "audio.ogg"  # важно для openai (расширение файла)

    transcription = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=bio,
        response_format="text",
        language="ru",
    )

    # В новых версиях transcription чаще всего строка
    if isinstance(transcription, str):
        return transcription.strip()

    text = getattr(transcription, "text", None)
    if text:
        return text.strip()

    return str(transcription).strip()


async def generate_daily_report(raw_text: str) -> str:
    """
    Генерация структурированного отчёта смены по черновику пользователя.
    """
    system_prompt = """
Ты — менеджер семейного ресторана, который каждый день пишет структурированный вечерний отчёт
для директора. Стиль деловой, спокойный, без лишних эмоций, но живой и понятный.

Всегда:
- сохраняй начало отчёта так, как прислал пользователь (обращение «Доброй ночи», дата, город);
- все цифры (гостей, магазин, городок, завтраки, купоны и т.п.) НЕ меняй, не придумывай новые;
- ниже сделай связный текстовый отчёт в 1–3 абзацах в том же стиле, как в предыдущих отчётах пользователя:
  • как прошёл день (спокойно, активно, равномерная посадка и т.д.),
  • банкетная нагрузка, брони, городок,
  • какие были жалобы/комментарии гостей и как их решили,
  • чем закончился день, общая оценка смены.

Используй только информацию из сообщения пользователя.
Не придумывай события, которых нет в тексте.
Если гость остался доволен после решения проблемы — подчеркни это.
"""
    resp = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        max_tokens=800,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# DB HELPERS: TASKS
# ---------------------------------------------------------------------------

def create_task(user_id: int, title: str, period: str) -> int:
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tasks (user_id, title, period, is_done, created_at, updated_at)
        VALUES (?, ?, ?, 0, ?, ?)
        """,
        (user_id, title, period, now, now),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()
    return task_id


def list_active_tasks(user_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, period
        FROM tasks
        WHERE user_id = ? AND is_done = 0
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def mark_task_done(task_id: int):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tasks SET is_done = 1, updated_at = ? WHERE id = ?",
        (now, task_id),
    )
    conn.commit()
    conn.close()


def delete_task(task_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# DB HELPERS: REPORTS
# ---------------------------------------------------------------------------

def save_report(user_id: int, raw_text: str, ai_text: str):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reports (user_id, created_at, raw_text, ai_text)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, now, raw_text, ai_text),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# HANDLERS
# ---------------------------------------------------------------------------

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    text = (
        "Привет! Это бот задач и отчётов.\n\n"
        "Я умею:\n"
        "• принимать задачи голосом или текстом (день / неделя / месяц);\n"
        "• показывать список активных задач с галочками и удалением;\n"
        "• собирать вечерний отчёт дня с помощью ИИ по твоему черновику.\n\n"
        "Выбери действие ниже:"
    )
    await message.answer(text, reply_markup=main_menu_keyboard())


# --- ЗАДАЧИ: старт добавления ------------------------------------------------

@dp.callback_query(F.data.startswith("add_task:"))
async def cb_add_task(callback: CallbackQuery, state: FSMContext):
    period = callback.data.split(":", maxsplit=1)[1]  # day/week/month
    await state.update_data(period=period)
    await state.set_state(TaskStates.waiting_for_task_content)

    period_label = {
        "day": "на сегодня",
        "week": "на эту неделю",
        "month": "на этот месяц",
    }.get(period, "")

    text = (
        f"Отправь голосовое или текст с задачей {period_label}.\n\n"
        "Пример голосом: «Сделать инвентарь текстиля»,\n"
        "Пример текстом: «Перепроверить городок перед закрытием»."
    )
    await callback.message.answer(text)
    await callback.answer()


# --- ЗАДАЧИ: приём текста/голоса ---------------------------------------------

@dp.message(TaskStates.waiting_for_task_content)
async def handle_new_task(message: Message, state: FSMContext):
    data = await state.get_data()
    period = data.get("period", "day")

    task_text: str | None = None

    if message.voice:
        await message.answer("Расшифровываю голос через Whisper…")
        voice_bytes_io = await bot.download(message.voice.file_id)
        voice_bytes = voice_bytes_io.read()
        try:
            task_text = await transcribe_voice(voice_bytes)
        except Exception as e:
            logging.exception("Ошибка транскрипции: %s", e)
            await message.answer("Не удалось расшифровать голос. Отправь, пожалуйста, текстом.")
            return
    elif message.text:
        task_text = message.text.strip()

    if not task_text:
        await message.answer("Не вижу текста задачи. Отправь голосовое или текст ещё раз.")
        return

    task_id = create_task(message.from_user.id, task_text, period)

    await state.clear()

    emoji = {"day": "📆", "week": "🗓", "month": "📅"}.get(period, "📝")
    await message.answer(
        f"{emoji} Задача сохранена:\n\n{task_text}",
        reply_markup=task_inline_keyboard(task_id),
    )


# --- ЗАДАЧИ: показать список --------------------------------------------------

@dp.callback_query(F.data == "show_tasks")
async def cb_show_tasks(callback: CallbackQuery):
    rows = list_active_tasks(callback.from_user.id)
    if not rows:
        await callback.message.answer("У тебя пока нет активных задач.")
        await callback.answer()
        return

    for task_id, title, period in rows:
        emoji = {"day": "📆", "week": "🗓", "month": "📅"}.get(period, "📝")
        text = f"{emoji} {title}"
        await callback.message.answer(text, reply_markup=task_inline_keyboard(task_id))

    await callback.answer()


# --- ЗАДАЧИ: обработка галочки/удаления --------------------------------------

@dp.callback_query(F.data.startswith("task_done:"))
async def cb_task_done(callback: CallbackQuery):
    try:
        task_id = int(callback.data.split(":", maxsplit=1)[1])
    except ValueError:
        await callback.answer("Ошибка ID задачи.", show_alert=True)
        return

    mark_task_done(task_id)
    await callback.answer("Задача отмечена выполненной ✅")

    try:
        old_text = callback.message.text or ""
        if "✅" not in old_text:
            new_text = old_text + " ✅"
            await callback.message.edit_text(new_text)
    except Exception:
        pass


@dp.callback_query(F.data.startswith("task_delete:"))
async def cb_task_delete(callback: CallbackQuery):
    try:
        task_id = int(callback.data.split(":", maxsplit=1)[1])
    except ValueError:
        await callback.answer("Ошибка ID задачи.", show_alert=True)
        return

    delete_task(task_id)
    await callback.answer("Задача удалена ❌")

    try:
        await callback.message.delete()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ОТЧЁТ ДНЯ С ИИ
# ---------------------------------------------------------------------------

@dp.callback_query(F.data == "daily_report_ai")
async def cb_daily_report_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(ReportStates.waiting_for_report_text)

    template = (
        "Отправь черновик отчёта одним сообщением в таком формате:\n\n"
        "Доброй ночи\n"
        "00.00.2025 Ташкент\n"
        "Гостей было: 00\n"
        "Магазин: 000.000\n"
        "Городок пробито: 00\n"
        "Городок записано: 00\n"
        "Не зашли: 0\n"
        "Завтрак: 0\n"
        "Купон: 0\n\n"
        "Важные моменты:\n"
        "- коротко пунктами опиши, что было важного за день\n"
        "- замечания гостей, отключения света, банкеты и т.п.\n\n"
        "Я соберу из этого профессиональный отчёт, как ты писал раньше."
    )

    await callback.message.answer(template)
    await callback.answer()


@dp.message(ReportStates.waiting_for_report_text)
async def handle_daily_report(message: Message, state: FSMContext):
    raw_text = message.text
    if not raw_text:
        await message.answer("Отправь, пожалуйста, отчёт текстом одним сообщением.")
        return

    await message.answer("Формирую отчёт с ИИ…")

    try:
        ai_text = await generate_daily_report(raw_text)
    except Exception as e:
        logging.exception("Ошибка генерации отчёта: %s", e)
        await message.answer("Не получилось сформировать отчёт, попробуй ещё раз чуть позже.")
        return

    await state.clear()

    save_report(message.from_user.id, raw_text, ai_text)

    await message.answer("Готовый отчёт дня:\n\n" + ai_text)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

async def main():
    init_db()
    logging.info("Бот задач и отчётов запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

