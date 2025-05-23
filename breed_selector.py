import json
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from text_utils import clean_text, correct_spelling_words, lemmatize
from voice_utils import recognize_voice_from_file, send_text_with_voice_button
from gtts import gTTS
import pickle
import subprocess

# Загружаем вопросы из JSON
with open("questions_about_breed.json", "r", encoding="utf-8") as f:
    questions = json.load(f)["questions_about_breed"]
    # Загрузка модели и векторизатора
with open("chatbot_model_for_breed_selector.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer_for_breed_selector.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Логика подбора породы
def select_cat_breed(answers):
    advertisement = "\nКошки, подходящие под ваш выбор, уже находятся у нас в зоомагазине и ищут тёплый, уютный дом. Может быть, один из них станет вашим верным любимцем."
    if answers["allergy"]:
        return "Сфинкс – отличная порода для аллергиков." + advertisement
    if answers["kids"] and answers["activity"]:
        return "Мейн-кун – игривая и дружелюбная кошка, отлично подходит для семьи." + advertisement
    if answers["care"]:
        return "Персидская кошка – требует ухода, но очень ласковая." + advertisement
    if answers["activity"] == "низкая":
        return "Британская короткошёрстная – спокойная и независимая." + advertisement
    if answers["fur_length"] == "без шерсти":
        return "Петерболд или сфинкс – отличный выбор для тех, кто предпочитает кошек без шерсти." + advertisement
    return "Рассмотрите русскую голубую – универсальная, не требует много ухода." + advertisement

# Старт диалога
async def start_breed_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["breed_dialog"] = {"step": 0, "answers": {}}
    first_question = questions[0]
    reply_markup = ReplyKeyboardMarkup(
        [[option] for option in first_question["options"]],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await update.message.reply_text(
        text=first_question["question"],
        reply_markup=reply_markup
)

# Обработка овпросов
async def handle_breed_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dialog = context.user_data.get("breed_dialog", {})
    step = dialog.get("step", 0)
    answers = dialog.get("answers", {})

    if step >= len(questions):
        await send_text_with_voice_button(update, context, "Что-то пошло не так. Попробуйте сначала.")
        context.user_data.pop("breed_dialog", None)
        return
    if update.message.voice:
        file = await update.message.voice.get_file()
        file_path = "voice.ogg"
        wav_path = "voice.wav"

        await file.download_to_drive(file_path)

        ffmpeg_path = r"C:\\Users\Aleksandra\\Study\\Intelligent Systems and Technologies 2\\Coursework\\CourseWork-ISAT\\ffmpeg-7.1.1-essentials_build\bin\\ffmpeg.exe"
        try:
            subprocess.run([ffmpeg_path, "-i", file_path, wav_path, "-y"], check=True)
        except Exception as e:
            await send_text_with_voice_button(update, context, "Ошибка! Попробуйте позже.")
            print("Ошибка при конвертации аудио")
            return

        user_input = recognize_voice_from_file(wav_path)

        if user_input:
            # Отправляем распознанный текст пользователю
            await send_text_with_voice_button(update, context, "Вы сказали: " + user_input)
        else:
            await send_text_with_voice_button(update, context, "Не удалось распознать голосовое сообщение.")
            return
    
    elif update.message.text:
        user_input = update.message.text.strip().lower()
    question = questions[step]
    key = question["key"]
    valid_options = question["options"]

    # Проверка на допустимый ответ
    cleaned = clean_text(user_input)
    pattern_to_tag = {option: option for option in valid_options}  # создаём словарь из списка
    corrected = correct_spelling_words(cleaned, pattern_to_tag)
    #corrected = correct_spelling_tag(cleaned, valid_options)
    # Лемматизация текста
    lemmatized_text = lemmatize(corrected)

    # Используем обученную модель
    if lemmatized_text not in valid_options:
        X = vectorizer.transform([lemmatized_text])
        predicted = model.predict(X)[0]
        
        # Проверим, попал ли ответ в options текущего вопроса
        if predicted in valid_options:
            lemmatized_text = predicted
        else:
            await send_text_with_voice_button(update, context, "Это не совсем то, что я ожидал. Попробуйте ещё раз:")
            await send_text_with_voice_button(update, context, f"{question['question']}\n(" + " / ".join(valid_options) + ")")
            return

    # Преобразование "да"/"нет" в bool
    if valid_options == ["да", "нет"]:
        answers[key] = corrected  == "да"
    else:
        answers[key] = corrected

    # Обновление состояния
    step += 1
    if step < len(questions):
        next_q = questions[step]
        reply_markup = ReplyKeyboardMarkup(
            [[option] for option in next_q["options"]],
            resize_keyboard=True,
            one_time_keyboard=True
        )
        context.user_data["breed_dialog"] = {"step": step, "answers": answers}
        await update.message.reply_text(
            text=next_q["question"],
            reply_markup=reply_markup
        )
    else:
        # Завершение диалога
        result = select_cat_breed(answers)
        # Удаляем клавиатуру (отдельным сообщением)
        await update.message.reply_text(
            "Спасибо за ответы!",
            reply_markup=ReplyKeyboardRemove()
        )
        # Теперь выводм сообщение с результатом
        await send_text_with_voice_button(update, context, f"Рекомендуемая порода: {result}")
        # Очищаем данные пользователя
        context.user_data.pop("breed_dialog", None)

