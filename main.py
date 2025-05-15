import os
import json
import random
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from breed_selector import handle_breed_dialog, start_breed_dialog
from text_utils import clean_text, correct_spelling_tag, lemmatize, correct_spelling_words
import pickle
import subprocess
from voice_utils import recognize_voice_from_file
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я твой усатый бот зоомагазина. Спросите что-нибудь!")

async def start_over(update: Update, context: ContextTypes.DEFAULT_TYPE, is_not_goodbye = True):
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
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.voice:
        file = await update.message.voice.get_file()
        file_path = "voice.ogg"
        wav_path = "voice.wav"

        await file.download_to_drive(file_path)

        # Корень проекта (файл main.py или аналог — от него и считается)
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
            await update.message.reply_text("Не удалось распознать голосовое сообщение.")
            return
    
    elif update.message.text:
        text = update.message.text.lower()

    # Если пользователь находится в диалоге подбора породы
    if "breed_dialog" in context.user_data:
        await handle_breed_dialog(update, context)
        return
    
    # Очищаем текст от лишних символов
    cleaned = clean_text(text)

    if cleaned in ["да", "ага", "хочу", "конечно", "угу"]:
        last = context.user_data.get("last_intent")
        print(last)
        if last == "питание_royal":
            print('Продолжение питание_royal')
            await update.message.reply_text("Вот ссылка на Royal Canin: https://example.com/royal-canin")
            context.user_data["last_intent"] = None
            return
        elif last == "питание_purina":
            print('Продолжение питание_purina')
            await update.message.reply_text("Вот ссылка на Purina One: https://example.com/purina")
            context.user_data["last_intent"] = None
            return
        elif last == "реклама":
            print('Продолжение рекламы')
            await update.message.reply_text("Подробности о скидках: https://example.com/sales")
            context.user_data["last_intent"] = None
            return
        elif last == "товары":
            print('Продолжение информации о товарах')
            await update.message.reply_text("Посмотрите предствленные в наличие товары: https://example.com/catalog")
            context.user_data["last_intent"] = None
            return

    # Создание списка допустимых вариантов из паттернов в intents.json
    pattern_to_tag = {}
    for intent in intents:
        for pattern in intent["patterns"]:
            pattern_to_tag[pattern] = intent["tag"]

    # Исправление опечаток, используя теги из intents.json
    corrected_tag, is_corrected = correct_spelling_tag(cleaned, pattern_to_tag)
    
    # Лемматизация текста
    #lemmatized_text = lemmatize(corrected)

    # Обработка интентов
    # Используем ML-модель для анализа намерения
    if is_corrected:
        predicted_tag = corrected_tag
    else:
        predicted_words = correct_spelling_words(cleaned, pattern_to_tag)
        # Получаем вероятности предсказания по всем тегам
        X = vectorizer.transform([predicted_words])  # Преобразуем текст в формат, который понимает модель
        probs = model.predict_proba(X)[0]  # Получаем вероятности для всех тегов
        max_prob_index = probs.argmax()  # Находим тег с максимальной вероятностью
        max_prob = probs[max_prob_index]
        predicted_tag = model.classes_[max_prob_index] # Извлекаем тег

        print(f"Predicted tag: {predicted_tag}, Probability: {max_prob}")
        # Уверенность предсказания должна быть хотя бы 0.6 (можно подстроить)
        if max_prob < 0.23:
            await update.message.reply_text("Я не совсем понял вас. Попробуйте переформулировать.")
            return
        
    tag_to_intent = {intent["tag"]: intent for intent in intents}
    # Если тег — это запуск подбора породы
    if predicted_tag == "подбор_породы":
        intent = tag_to_intent[predicted_tag]
        response = random.choice(intent["responses"])
        await update.message.reply_text(response)
        await start_breed_dialog(update, context)
        return
    elif predicted_tag == "прощание":
        intent = tag_to_intent[predicted_tag]
        response = random.choice(intent["responses"])
        await update.message.reply_text(response)
        await start_over(update, context, False)
        return
    elif predicted_tag in tag_to_intent:
        intent = tag_to_intent[predicted_tag]
        response = random.choice(intent["responses"])
        await update.message.reply_text(response)

        # Если этот ответ бота требует подтверждения — сохранить last_intent
        if predicted_tag in ["питание_royal", "питание_purina", "скидки", "товары", "реклама"]:
            context.user_data["last_intent"] = predicted_tag
            print(f"Сохраняем last_intent: {predicted_tag}")

        return
        
    # Если ничего не подошло
    await update.message.reply_text("Не понял вопрос. Попробуйте иначе.")

def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("start_over", start_over))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.VOICE, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
