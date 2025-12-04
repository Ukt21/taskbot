import asyncio
import json
import tempfile
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

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
from aiogram import Dispatcher

dp = Dispatcher()
dp.include_router(router_report)
# dp.include_router(router_tasks)
# dp.include_router(router_other)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import get_settings
from prompts import TASK_ASSISTANT_SYSTEM_PROMPT
from db import (
    init_db,
    add_task,
    get_tasks,
    get_done_tasks,
    set_task_done,
    update_task_title,
    delete_task,
    get_all_user_ids,
    get_daily_summary,
    get_stats,
    get_last_report_date,
    update_last_report_date,
)


# ---------------- ИНИЦИАЛИЗАЦИЯ ----------------

settings = get_settings()
bot = Bot(token=settings.bot_token)
client = OpenAI(api_key=settings.openai_api_key)

storage = MemoryStorage()
dp = Dispatcher(storage=storage)

PERIOD_LABELS_RU = {
    "day": "сегодня",
    "week": "на этой неделе",
    "month": "в этом месяце",
    "all": "за все время",
    "auto": "период",
}

PRIORITY_ICONS = {
    3: "🔴",
    2: "🟠",
    1: "🟢",
    0: "⚪️",
}

import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_daily_report(raw_text: str) -> str:
    system_prompt = """
Ты — менеджер ресторана, который каждый день пишет структурированный вечерний отчёт
для директора. Тон деловой, спокойный, без лишних эмоций, но живой и понятный.

Всегда:
- сохраняй начало так, как прислал пользователь: обращение «Доброй ночи», дата, город;
- цифры (гостей, магазин, городок, завтраки, купоны и т.п.) НЕ меняй, только аккуратно
  опиши их словами;
- после блока цифр сделай связный текстовый отчёт в 1–3 абзацах в том же стиле,
  как в прошлых отчётах пользователя:
  - как прошёл день (спокойно, активно, большая посадка и т.д.),
  - что было по банкетам, броням, нагрузке;
  - какие были жалобы/комментарии гостей и как их решили;
  - чем закончился день, общая оценка.

Не придумывай события, опирайся только на важные моменты из текста пользователя.
Если гость остался доволен после решения проблемы, обязательно это укажи.
"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
        max_tokens=800,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()
    from aiogram import Router, F
from aiogram.types import CallbackQuery, Message
from aiogram.fsm.context import FSMContext

router = Router()

@router.callback_query(F.data == "daily_report_ai")
async def start_daily_report(callback: CallbackQuery, state: FSMContext):
    await state.set_state(ReportStates.waiting_for_report_text)

    template = (
        "Скопируй, заполни цифры и добавь важные моменты дня одним сообщением.\n\n"
        "Пример:\n\n"
        "Доброй ночи\n"
        "01.01.2025 Ташкент\n"
        "Гостей было: 85\n"
        "Магазин: 123.450\n"
        "Городок пробито: 35\n"
        "Городок записано: 28\n"
        "Не зашли: 2\n"
        "Завтрак: 5\n"
        "Купон: 3\n\n"
        "Важные моменты:\n"
        "- было отключение света в Сити Молл;\n"
        "- один гость пожаловался на прожарку стейка, сделали замену;\n"
        "- банкет на 20 персон прошёл без замечаний.\n\n"
        "Отправь свой вариант, а я сделаю из него структурированный отчёт."
    )

    await callback.message.answer(template)
    await callback.answer()

# ---------------- КЛАВИАТУРЫ ----------------
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

main_menu = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="📊 Отчёт дня с ИИ", callback_data="daily_report_ai")],
    # здесь твои остальные кнопки
])
def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="➕ Задача на день",
                    callback_data="add:day",
                )
            ],
            [
                InlineKeyboardButton(
                    text="➕ Задача на неделю",
                    callback_data="add:week",
                )
            ],
            [
                InlineKeyboardButton(
                    text="➕ Задача на месяц",
                    callback_data="add:month",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📊 Отчёт по задачам",
                    callback_data="report_menu",
                )
            ],
        ]
    )


def report_menu_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📅 На сегодня", callback_data="report_period:day")
    kb.button(text="📆 На неделю", callback_data="report_period:week")
    kb.button(text="🗓 На месяц", callback_data="report_period:month")
    kb.button(text="📋 Все активные", callback_data="report_period:all")
    kb.button(text="✅ Выполненные", callback_data="report_done:all")
    kb.button(text="📜 История", callback_data="history")
    kb.adjust(2, 2, 2)
    return kb.as_markup()


def build_task_buttons(user_id: int, period: Optional[str], done: bool = False):
    """
    Строим компактную «карточную» клавиатуру задач.
    period:
      - 'day' / 'week' / 'month' / 'all'
      - None → все активные
    done:
      - False → активные
      - True → выполненные
    """
    if done:
        tasks = get_done_tasks(user_id, period)
    else:
        tasks = get_tasks(user_id, period, only_active=True)

    if not tasks:
        return None

    kb = InlineKeyboardBuilder()
    for t in tasks:
        priority = t.get("priority") or 0
        icon = PRIORITY_ICONS.get(priority, "⚪️")
        title = t["title"]
        raw_deadline = t.get("raw_deadline") or ""
        extra = f" • {raw_deadline}" if raw_deadline else ""
        text = f"{icon} {title}{extra}"

        # Строка: [✅ ...] [✏️] [❌]
        kb.button(text=f"✅ {title}", callback_data=f"done:{t['id']}")
        kb.button(text="✏️", callback_data=f"edit:{t['id']}")
        kb.adjust(2)

        return kb.as_markup()
        

# ---------------- РАСШИФРОВКА ГОЛОСА (WHISPER) ----------------

async def transcribe_voice(message: Message) -> str:
    tmp_path = Path(tempfile.gettempdir()) / f"voice_{message.message_id}.oga"

    tg_file = await bot.get_file(message.voice.file_id)
    await bot.download_file(tg_file.file_path, tmp_path)

    try:
        with tmp_path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text",
                # language="ru",
            )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result


# ---------------- ВЫЗОВ ИИ ДЛЯ РАЗБОРА ЗАДАЧ ----------------

async def call_task_model(button: str, period: str, text: str) -> Dict[str, Any]:
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
                "model": settings.openai_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


# ---------------- ОЦЕНКА ПРИОРИТЕТА / КАТЕГОРИИ ----------------

def infer_priority(raw_deadline: str, description: str) -> int:
    text = f"{raw_deadline} {description}".lower()
    if any(x in text for x in ["срочно", "прямо сейчас", "как можно быстрее", "до конца дня", "сегодня"]):
        return 3
    if any(x in text for x in ["завтра", "на этой неделе", "до завтра"]):
        return 2
    return 1


def infer_category(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    work_words = ["банкет", "гость", "отчёт", "выручка", "смена", "официант", "кухня", "сотрудник", "график"]
    home_words = ["дом", "ребёнок", "ребенка", "магазин", "купить", "семья", "уборка"]
    if any(w in text for w in work_words):
        return "work"
    if any(w in text for w in home_words):
        return "home"
    return "other"


# ---------------- СОСТОЯНИЯ FSM ----------------

class AddTaskState(StatesGroup):
    waiting_voice_or_text = State()


class EditTaskState(StatesGroup):
    waiting_new_title = State()
from aiogram.fsm.state import StatesGroup, State

class ReportStates(StatesGroup):
    waiting_for_report_text = State()

# ---------------- /start ----------------

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот для управления задачами по голосу.\n"
        "Добавляй задачи на день, неделю, месяц и отмечай выполнение кнопками.",
        reply_markup=main_menu_keyboard(),
    )


# ---------------- ДОБАВЛЕНИЕ ЗАДАЧ ----------------

@dp.callback_query(F.data.startswith("add:"))
async def callback_add(callback: CallbackQuery, state: FSMContext):
    _, period = callback.data.split(":", maxsplit=1)
    ru = PERIOD_LABELS_RU.get(period, period)

    await state.set_state(AddTaskState.waiting_voice_or_text)
    await state.update_data(period=period)

    await callback.message.answer(
        f"Отправь голосовое или текст с задачами {ru}.\n"
        f"Я сам их распознаю и сохраню.",
        reply_markup=main_menu_keyboard(),
    )
    await callback.answer()


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
    try:
        result = await call_task_model("add", period, text)
    except Exception as e:
        await message.answer(f"Ошибка ИИ: {e}")
        return

    user_id = message.from_user.id
    tasks_list = result.get("tasks", [])

    for item in tasks_list:
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        raw_deadline = (item.get("raw_deadline") or "").strip()
        if not title:
            continue

        priority = infer_priority(raw_deadline, description)
        category = infer_category(title, description)

        add_task(
            user_id=user_id,
            title=title,
            period=period,
            description=description or None,
            raw_deadline=raw_deadline or None,
            category=category,
            priority=priority,
        )

    kb = build_task_buttons(user_id, period, done=False)
    if kb:
        await message.answer(
            f"Добавлено задач: {len(tasks_list)}",
            reply_markup=kb,
        )
    else:
        await message.answer("Не удалось выделить задачи из текста.")


# ---------------- ОТЧЁТ ПО ЗАДАЧАМ ----------------

@dp.callback_query(F.data == "report_menu")
async def cb_report_menu(callback: CallbackQuery):
    await callback.message.answer(
        "Выбери период для отчёта:",
        reply_markup=report_menu_keyboard(),
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("report_period:"))
async def cb_report_period(callback: CallbackQuery):
    _, period = callback.data.split(":", maxsplit=1)
    user_id = callback.from_user.id

    kb = build_task_buttons(user_id, period if period != "all" else None, done=False)
    period_text = PERIOD_LABELS_RU.get(period, period)

    if kb:
        await callback.message.answer(
            f"Активные задачи {period_text}:",
            reply_markup=kb,
        )
    else:
        await callback.message.answer(f"Нет активных задач {period_text}.")
    await callback.answer()


@dp.callback_query(F.data.startswith("report_done:"))
async def cb_report_done(callback: CallbackQuery):
    user_id = callback.from_user.id
    kb = build_task_buttons(user_id, None, done=True)

    if kb:
        await callback.message.answer(
            "Выполненные задачи:",
            reply_markup=kb,
        )
    else:
        await callback.message.answer("Пока нет выполненных задач.")
    await callback.answer()


@dp.callback_query(F.data == "history")
async def cb_history(callback: CallbackQuery):
    user_id = callback.from_user.id
    stats = get_stats(user_id)

    text = (
        "📜 История задач:\n"
        f"Всего задач: {stats['total']}\n"
        f"Выполнено: {stats['done']}\n"
        f"Активных: {stats['active']}"
    )
    await callback.message.answer(text)
    await callback.answer()


# ---------------- ОБРАБОТКА КНОПОК ✅ / ❌ / ✏️ ----------------

@dp.callback_query(F.data.startswith("done:"))
async def cb_done(callback: CallbackQuery):
    task_id = int(callback.data.split(":", maxsplit=1)[1])
    set_task_done(task_id)
    await callback.answer("Задача выполнена ✅")
    await refresh_after_change(callback)


@dp.callback_query(F.data.startswith("delete:"))
async def cb_delete(callback: CallbackQuery):
    task_id = int(callback.data.split(":", maxsplit=1)[1])
    delete_task(task_id)
    await callback.answer("Задача удалена ❌")
    await refresh_after_change(callback)


class EditTaskState(StatesGroup):
    waiting_new_title = State()


@dp.callback_query(F.data.startswith("edit:"))
async def cb_edit(callback: CallbackQuery, state: FSMContext):
    task_id = int(callback.data.split(":", maxsplit=1)[1])
    await state.set_state(EditTaskState.waiting_new_title)
    await state.update_data(task_id=task_id)

    await callback.message.answer(
        "Отправь новый текст задачи (можно голосом или текстом).",
    )
    await callback.answer()


@dp.message(EditTaskState.waiting_new_title, F.voice)
async def edit_task_voice(message: Message, state: FSMContext):
    data = await state.get_data()
    task_id = data["task_id"]
    try:
        text = await transcribe_voice(message)
    except Exception as e:
        await message.answer(f"Ошибка STT при редактировании: {e}")
        return

    new_title = text.strip()
    if not new_title:
        await message.answer("Текст пустой, задача не изменена.")
        await state.clear()
        return

    update_task_title(task_id, new_title)
    await message.answer("Задача обновлена.")
    await state.clear()


@dp.message(EditTaskState.waiting_new_title, F.text)
async def edit_task_text(message: Message, state: FSMContext):
    data = await state.get_data()
    task_id = data["task_id"]

    new_title = message.text.strip()
    if not new_title:
        await message.answer("Текст пустой, задача не изменена.")
        await state.clear()
        return

    update_task_title(task_id, new_title)
    await message.answer("Задача обновлена.")
    await state.clear()


async def refresh_after_change(callback: CallbackQuery):
    user_id = callback.from_user.id
    # обновляем последнюю клавиатуру, исходя из того, что там были активные задачи
    kb = build_task_buttons(user_id, None, done=False)
    if kb:
        await callback.message.edit_reply_markup(reply_markup=kb)
    else:
        await callback.message.edit_text("🎉 Все задачи выполнены!", reply_markup=None)


# ---------------- ЕЖЕДНЕВНЫЙ АВТО-ОТЧЁТ В 21:00 ----------------

async def daily_report_worker():
    """
    Раз в минуту проверяем время.
    В 21:00 по серверному времени отправляем авто-отчёт пользователям,
    у которых есть задачи, и помечаем в daily_reports, что за этот день отчёт отправлен.
    """
    while True:
        now = datetime.datetime.now()
        if now.hour == 21 and now.minute == 0:
            today_str = now.date().isoformat()
            user_ids = get_all_user_ids()
            for user_id in user_ids:
                last_date = get_last_report_date(user_id)
                if last_date == today_str:
                    continue  # уже отправляли

                summary = get_daily_summary(user_id)
                text = (
                    "📊 Итоги дня:\n"
                    f"Выполнено сегодня: {summary['done_today']}\n"
                    "Активные задачи:\n"
                    f"  • Сегодня: {summary['active']['day']}\n"
                    f"  • Неделя: {summary['active']['week']}\n"
                    f"  • Месяц: {summary['active']['month']}"
                )
                try:
                    await bot.send_message(chat_id=user_id, text=text)
                except Exception:
                    # если пользователь заблокировал бота или ошибка — просто игнорируем
                    pass

                update_last_report_date(user_id, today_str)

            # чтобы не спамить в ту же минуту, чуть ждём
            await asyncio.sleep(65)
        else:
            await asyncio.sleep(30)


# ---------------- MAIN ----------------

async def main():
    init_db()
    asyncio.create_task(daily_report_worker())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
