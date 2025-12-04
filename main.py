import asyncio
import logging
import os
import sqlite3
from io import BytesIO

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from openai import AsyncOpenAI

# ================== НАСТРОЙКИ ==================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения")
if not OPENAI_API_KEY:
    raise RuntimeError("Не задан OPENAI_API_KEY в переменных окружения")

bot = Bot(
    token=TELEGRAM_BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher(storage=MemoryStorage())
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)

# ================== SQLITE ==================

DB_PATH = "tasks.db"


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            scope TEXT NOT NULL,          -- day/week/month
            done INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


db = init_db()


def add_task(user_id: int, text: str, scope: str) -> int:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO tasks (user_id, text, scope, done) VALUES (?, ?, ?, 0)",
        (user_id, text.strip(), scope),
    )
    db.commit()
    return cur.lastrowid


def list_tasks(user_id: int):
    cur = db.cursor()
    cur.execute(
        "SELECT id, text, scope, done FROM tasks WHERE user_id = ? ORDER BY done, id",
        (user_id,),
    )
    return cur.fetchall()


def set_task_done(task_id: int, done: bool):
    cur = db.cursor()
    cur.execute("UPDATE tasks SET done = ? WHERE id = ?", (1 if done else 0, task_id))
    db.commit()


def delete_task(task_id: int):
    cur = db.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    db.commit()


# ================== ШАБЛОН ОТЧЁТА ==================

REPORT_HEADER_TEMPLATE = """Доброй ночи
00.00.2025 Ташкент
Гостей было: 00
Магазин: 000.000
Городок пробито: 00
Городок записано: 00
Не зашли: 0
Завтрак: 0
Купон: 0
"""

# ================== СОСТОЯНИЯ FSM ==================


class AddTaskState(StatesGroup):
    waiting_for_text = State()


class ReportState(StatesGroup):
    waiting_for_points = State()


# ================== КЛАВИАТУРЫ ==================


def main_menu_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Задача на день", callback_data="add_task:day")
    kb.button(text="➕ Задача на неделю", callback_data="add_task:week")
    kb.button(text="➕ Задача на месяц", callback_data="add_task:month")
    kb.button(text="📋 Мои задачи", callback_data="list_tasks")
    kb.button(text="📝 Отчёт с ИИ", callback_data="daily_report")
    kb.adjust(1)
    return kb.as_markup()


def tasks_kb(tasks_rows) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for row in tasks_rows:
        task_id = row["id"]
        text = row["text"]
        done = bool(row["done"])
        status = "✅" if done else "⬜️"
        caption = (text[:40] + "…") if len(text) > 43 else text

        kb.row(
            InlineKeyboardButton(
                text=f"{status} {caption}", callback_data="noop"
            ),
        )
        kb.row(
            InlineKeyboardButton(
                text="✔ Выполнено", callback_data=f"task_done:{task_id}"
            ),
            InlineKeyboardButton(
                text="✖ Удалить", callback_data=f"task_delete:{task_id}"
            ),
        )
    kb.row(InlineKeyboardButton(text="⬅ В меню", callback_data="back_to_menu"))
    return kb.as_markup()


# ================== УТИЛИТЫ ==================


async def transcribe_voice(message: Message) -> str | None:
    """
    Расшифровка voice через Whisper.
    Возвращает текст или None, если не удалось.
    """
    try:
        voice = message.voice or message.audio
        if not voice:
            return None

        file = await bot.get_file(voice.file_id)
        byte_io: BytesIO = await bot.download_file(file.file_path)
        byte_io.name = "audio.ogg"  # нужно имя файла для OpenAI SDK

        transcription = await openai_client.audio.transcriptions.create(
            model="whisper-1",  # модель Whisper
            file=byte_io,
            language="ru",
        )
        # у Whisper ответ в поле text
        text = transcription.text.strip()
        return text or None
    except Exception as e:
        logging.exception("Ошибка при расшифровке голоса: %s", e)
        return None


async def generate_report_text(points: str) -> str:
    """
    Генерируем тело отчёта (без шапки) через GPT-4o-mini.
    """
    system_prompt = (
        "Ты помощник управляющего рестораном. "
        "На основе краткого описания дня напиши один профессиональный, "
        "структурированный отчёт смены. Не пиши приветствия, даты и города – "
        "сразу переходи к описанию дня, загрузки, банкетов, жалоб и выводов. "
        "Пиши от первого лица единственного числа."
    )

    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Краткое описание дня:\n{points.strip()}",
            },
        ],
        temperature=0.4,
    )
    body = completion.choices[0].message.content.strip()
    return body


# ================== ХЕНДЛЕРЫ ==================


@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    text = (
        "Привет, это бот задач и отчётов.\n\n"
        "Ты можешь:\n"
        "• добавлять задачи на день, неделю и месяц (в том числе голосом);\n"
        "• отмечать задачи выполненными;\n"
        "• формировать отчёт смены с помощью ИИ.\n\n"
        "Выбери действие:"
    )
    await message.answer(text, reply_markup=main_menu_kb())


@dp.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Главное меню:", reply_markup=main_menu_kb())


# ---------- Добавление задач ----------


@dp.callback_query(F.data.startswith("add_task:"))
async def cb_add_task(call: CallbackQuery, state: FSMContext):
    scope = call.data.split(":", maxsplit=1)[1]  # day/week/month
    await state.set_state(AddTaskState.waiting_for_text)
    await state.update_data(scope=scope)
    scope_label = {
        "day": "день",
        "week": "неделю",
        "month": "месяц",
    }.get(scope, "день")

    await call.message.answer(
        f"Продиктуй или напиши текст задачи на <b>{scope_label}</b>.\n"
        "Можно отправить голосовое сообщение.",
    )
    await call.answer()


@dp.message(AddTaskState.waiting_for_text, F.voice)
async def add_task_from_voice(message: Message, state: FSMContext):
    data = await state.get_data()
    scope = data.get("scope", "day")

    text = await transcribe_voice(message)
    if not text:
        await message.answer(
            "Не удалось расшифровать голос. Пожалуйста, отправь текстом задачу."
        )
        return

    task_id = add_task(message.from_user.id, text, scope)
    await message.answer(
        f"Задача добавлена (ID {task_id}):\n• {text}",
        reply_markup=main_menu_kb(),
    )
    await state.clear()


@dp.message(AddTaskState.waiting_for_text, F.text)
async def add_task_from_text(message: Message, state: FSMContext):
    data = await state.get_data()
    scope = data.get("scope", "day")
    text = message.text.strip()
    if not text:
        await message.answer("Пустая задача, отправь нормальный текст.")
        return

    task_id = add_task(message.from_user.id, text, scope)
    await message.answer(
        f"Задача добавлена (ID {task_id}):\n• {text}",
        reply_markup=main_menu_kb(),
    )
    await state.clear()


# ---------- Список задач ----------


@dp.callback_query(F.data == "list_tasks")
async def cb_list_tasks(call: CallbackQuery, state: FSMContext):
    await state.clear()
    rows = list_tasks(call.from_user.id)
    if not rows:
        await call.message.answer(
            "У тебя пока нет задач.", reply_markup=main_menu_kb()
        )
        await call.answer()
        return

    await call.message.answer(
        "Текущие задачи:", reply_markup=tasks_kb(rows)
    )
    await call.answer()


@dp.callback_query(F.data == "back_to_menu")
async def cb_back_to_menu(call: CallbackQuery, state: FSMContext):
    await state.clear()
    await call.message.answer("Главное меню:", reply_markup=main_menu_kb())
    await call.answer()


@dp.callback_query(F.data.startswith("task_done:"))
async def cb_task_done(call: CallbackQuery):
    try:
        task_id = int(call.data.split(":", maxsplit=1)[1])
    except ValueError:
        await call.answer("Ошибка ID задачи", show_alert=True)
        return

    set_task_done(task_id, True)
    rows = list_tasks(call.from_user.id)
    text = "Задача отмечена как выполненная."
    if rows:
        await call.message.edit_text(text + "\n\nТекущие задачи:", reply_markup=tasks_kb(rows))
    else:
        await call.message.edit_text(text)
        await call.message.answer("Задач больше нет.", reply_markup=main_menu_kb())
    await call.answer("Готово")


@dp.callback_query(F.data.startswith("task_delete:"))
async def cb_task_delete(call: CallbackQuery):
    try:
        task_id = int(call.data.split(":", maxsplit=1)[1])
    except ValueError:
        await call.answer("Ошибка ID задачи", show_alert=True)
        return

    delete_task(task_id)
    rows = list_tasks(call.from_user.id)
    text = "Задача удалена."
    if rows:
        await call.message.edit_text(text + "\n\nТекущие задачи:", reply_markup=tasks_kb(rows))
    else:
        await call.message.edit_text(text)
        await call.message.answer("Задач больше нет.", reply_markup=main_menu_kb())
    await call.answer("Удалено")


# ---------- Отчёт с ИИ ----------


@dp.callback_query(F.data == "daily_report")
async def cb_daily_report(call: CallbackQuery, state: FSMContext):
    await state.set_state(ReportState.waiting_for_points)

    header_text = (
        f"{REPORT_HEADER_TEMPLATE}\n"
        "Важные моменты:\n"
        "- коротко пунктами опиши, что было важного за день;\n"
        "- замечания гостей, отключения света, банкеты и т.п.\n\n"
        "Отправь голосом или текстом важные моменты дня, "
        "я соберу из этого профессиональный отчёт."
    )

    await call.message.answer(header_text)
    await call.answer()


@dp.message(ReportState.waiting_for_points, F.voice)
async def report_points_voice(message: Message, state: FSMContext):
    points = await transcribe_voice(message)
    if not points:
        await message.answer(
            "Не удалось расшифровать голос. Пожалуйста, отправь важные моменты текстом."
        )
        return

    await _finish_report(message, points, state)


@dp.message(ReportState.waiting_for_points, F.text)
async def report_points_text(message: Message, state: FSMContext):
    points = message.text.strip()
    if not points:
        await message.answer("Опиши, пожалуйста, как прошёл день.")
        return

    await _finish_report(message, points, state)


async def _finish_report(message: Message, points: str, state: FSMContext):
    await message.answer("Формирую отчёт с ИИ…")
    try:
        body = await generate_report_text(points)
        final_report = f"{REPORT_HEADER_TEMPLATE}\n{body}"

        await message.answer(
            f"<b>Готовый отчёт дня:</b>\n\n{final_report}"
        )
    except Exception as e:
        logging.exception("Ошибка генерации отчёта: %s", e)
        await message.answer(
            "Не получилось сформировать отчёт. Попробуй ещё раз чуть позже."
        )
    finally:
        await state.clear()


# ---------- NOOP для строк задач ----------


@dp.callback_query(F.data == "noop")
async def cb_noop(call: CallbackQuery):
    # Ничего не делаем, чтобы нажимать на строку задачи без ошибки
    await call.answer()


# ================== ЗАПУСК ==================


async def main():
    logging.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        db.close()
