import os
import json
import random
from pathlib import Path
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from breed_selector import handle_breed_dialog, start_breed_dialog
from text_utils import clean_text, correct_spelling_tag, correct_spelling_words
import pickle
import subprocess
from voice_utils import recognize_voice_from_file, send_text_with_voice_button, text_to_voice_by_clicking_button
from dotenv import load_dotenv

load_dotenv(dotenv_path="token.env")

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    intents = data['intents']
# Загрузка модели и векторизатора
with open("chatbot_model_main.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer_main.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Загружаем вопросы из JSON
with open("answers_for_advertisement.json", "r", encoding="utf-8") as f:
    answers = json.load(f)["answers_for_advertisement"]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_text_with_voice_button(update, context, "Привет! Я твой усатый бот зоомагазина. Спросите что-нибудь!")

async def start_over(update: Update, context: ContextTypes.DEFAULT_TYPE, is_not_goodbye = True):
    # Удаляем обычную клавиатуру
    await update.message.reply_text(
        "Перезагружаю своё сознание...",
        reply_markup=ReplyKeyboardRemove()
    )
    # Сброс данных опроса
    context.user_data.pop("breed_dialog", None)
    if is_not_goodbye:
        await start(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Привет! Я твой усатый бот зоомагазина. 🐾\n"
        "Можешь меня спросить:\n"
        "- о скидках\n"
        "- о здоровье питомца\n"
        "- о интересных предложениях\n"
        "- о кормах и других товарах\n"
        "- также я могу помочь выбрать тебе породу котика\n\n"
        "Также я знаю некоторые команды. Введи:\n"
        "/start - Чтобы запустить меня, усатого бота\n"
        "/start_over - Чтобы сбросить прогресс диалога и начинается заново\n"
        "/help - Чтобы показать это сообщение"
    )
    await send_text_with_voice_button(update, context, help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.voice:
        file = await update.message.voice.get_file()
        file_path = "voice.ogg"
        wav_path = "voice.wav"

        await file.download_to_drive(file_path)

        # Корень проекта (там, где файл main.py)
        BASE_DIR = Path(__file__).resolve().parent
        # Относительный путь к ffmpeg
        ffmpeg_path = str(BASE_DIR / "ffmpeg-7.1.1-essentials_build" / "bin" / "ffmpeg.exe")
        
        try:
            subprocess.run([ffmpeg_path, "-i", file_path, wav_path, "-y"], check=True)
        except Exception as e:
            await update.message.reply_text("Ошибка! Попробуйте позже.")
            print("Ошибка при конвертации аудио")
            return

        text = recognize_voice_from_file(wav_path)

        if text:
            # Отправляем распознанный текст пользователю
            await update.message.reply_text("Вы сказали: " + text)
        else:
            await send_text_with_voice_button(update, context, "Не удалось распознать голосовое сообщение.")
            return
    
    elif update.message.text:
        text = update.message.text.lower()

    # Если пользователь находится в диалоге подбора породы
    if "breed_dialog" in context.user_data:
        await handle_breed_dialog(update, context)
        return
    
    # Очищаем текст от лишних символов
    cleaned = clean_text(text)
    
    # Создание списка допустимых вариантов из паттернов в intents.json
    pattern_to_tag = {}
    for intent in intents:
        for pattern in intent["patterns"]:
            pattern_to_tag[pattern] = intent["tag"]

    # Исправление опечаток, используя теги из intents.json
    corrected_tag, is_corrected = correct_spelling_tag(cleaned, pattern_to_tag)

    # Обработка интентов
    # Используем ML-модель для анализа намерения
    if is_corrected:
        predicted_tag = corrected_tag
    else:
        predicted_words = correct_spelling_words(cleaned, pattern_to_tag)
        # Получаем вероятности предсказания по всем тегам
        X = vectorizer.transform([predicted_words])  # Преобразование текста в вектор
        probs = model.predict_proba(X)[0]  # Получение вероятностей для всех тегов
        max_prob_index = probs.argmax()  # Тег с максимальной вероятностью
        max_prob = probs[max_prob_index] # Получение этой максимальной вероятностьи наиболее вероятного тэга
        predicted_tag = model.classes_[max_prob_index] # Извлекаем тег

        print(f"Predicted tag: {predicted_tag}, Probability: {max_prob}")
        if max_prob < 0.23:
            await send_text_with_voice_button(update, context, "Я не совсем понял вас. Попробуйте переформулировать.")
            return
        
    tag_to_intent = {intent["tag"]: intent for intent in intents}
    # Если тег — это запуск подбора породы
    
    last = context.user_data.get("last_intent")
    continue_advertisement_intents = ["питание_royal", "питание_purina", "реклама", "товары"]
    if last in continue_advertisement_intents:
        tag_to_intent_advertisement = {answer["tag"]: answer for answer in answers}
        if predicted_tag == "согласие":
            last = context.user_data.get("last_intent")
            intent = tag_to_intent_advertisement[predicted_tag]
            if last == "питание_royal":
                await send_text_with_voice_button(update, context, intent["responses"][0])
                context.user_data["last_intent"] = None
                return
            elif last == "питание_purina":
                await send_text_with_voice_button(update, context, intent["responses"][1])
                context.user_data["last_intent"] = None
                return
            elif last == "реклама":
                await send_text_with_voice_button(update, context, intent["responses"][2])
                context.user_data["last_intent"] = None
                return
            elif last == "товары":
                await send_text_with_voice_button(update, context, intent["responses"][3])
                context.user_data["last_intent"] = None
                return
        elif predicted_tag == "отрицание":
            intent = tag_to_intent_advertisement[predicted_tag]
            response = random.choice(intent["responses"])
            await send_text_with_voice_button(update, context, response)
            context.user_data["last_intent"] = None
            return
        else:
            # Повторно обрабатываем такой же интент
            intent = tag_to_intent[predicted_tag]
            response = random.choice(intent["responses"])
            await send_text_with_voice_button(update, context, response)
            context.user_data["last_intent"] = None
            return

    elif predicted_tag == "подбор_породы":
        intent = tag_to_intent[predicted_tag]
        response = random.choice(intent["responses"])
        await send_text_with_voice_button(update, context, response)
        await start_breed_dialog(update, context)
        return
    elif predicted_tag == "прощание":
        intent = tag_to_intent[predicted_tag]
        response = random.choice(intent["responses"])
        await send_text_with_voice_button(update, context, response)
        await start_over(update, context, False)
        return
    elif predicted_tag in tag_to_intent:
        intent = tag_to_intent[predicted_tag]
        response = random.choice(intent["responses"])
        await send_text_with_voice_button(update, context, response)

        # Если этот ответ бота требует подтверждения — сохранить last_intent
        if predicted_tag in continue_advertisement_intents:
            context.user_data["last_intent"] = predicted_tag
            print(f"Сохраняем last_intent: {predicted_tag}")

        return
        
    # Если ничего не подошло
    await send_text_with_voice_button(update, context, "Не понял вопрос. Попробуйте иначе.")

def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("start_over", start_over))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.VOICE, handle_message))
    app.add_handler(CallbackQueryHandler(text_to_voice_by_clicking_button))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
