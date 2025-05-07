import json
import random
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from breed_selector import handle_breed_dialog, start_breed_dialog
from text_utils import clean_text, correct_spelling_tag, lemmatize, correct_spelling_words
import pickle


with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    intents = data['intents']
# Загрузка модели и векторизатора
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


async def start(update: Update, context):
    await update.message.reply_text("Привет! Я бот зоомагазина. Спросите что-нибудь!")

async def handle_message(update: Update, context):
    text = update.message.text.lower()

    # Если пользователь находится в диалоге подбора породы
    if "breed_dialog" in context.user_data:
        await handle_breed_dialog(update, context)
        return
    
    # Очищаем текст от лишних символов
    cleaned = clean_text(text)

    # Создание списка допустимых вариантов из тегов в intents.json
    #valid_options = [intent["tag"] for intent in intents]

    pattern_to_tag = {}
    for intent in intents:
        for pattern in intent["patterns"]:
            pattern_to_tag[pattern] = intent["tag"]

    # Исправление опечаток, используя теги из intents.json
    predicted_tag, corrected = correct_spelling_tag(cleaned, pattern_to_tag)
    
    # Лемматизация текста
    #lemmatized_text = lemmatize(corrected)

    # Обработка интентов
    # Используем ML-модель для анализа намерения
    
    if corrected:
        tag_to_intent = {intent["tag"]: intent for intent in intents}
        # Если тег — это запуск подбора породы
        if predicted_tag == "подбор_породы":
            await start_breed_dialog(update, context)
            return
        else:
            if predicted_tag in tag_to_intent:
                intent = tag_to_intent[predicted_tag]
                response = random.choice(intent["responses"])
                await update.message.reply_text(response)
                return
    else:
        predicted_words = correct_spelling_words(cleaned, pattern_to_tag)
        X = vectorizer.transform([predicted_words])
        # Получаем вероятности предсказания по всем тегам
        probs = model.predict_proba(X)[0]
        max_prob_index = probs.argmax()
        max_prob = probs[max_prob_index]
        predicted_tag_second = model.classes_[max_prob_index]

        print(f"Predicted tag: {predicted_tag_second}, Probability: {max_prob}")
        # Уверенность предсказания должна быть хотя бы 0.6 (можно подстроить)
        if max_prob < 0.23:
            await update.message.reply_text("Я не совсем понял вас. Попробуйте переформулировать.")
            return
        
        tag_to_intent = {intent["tag"]: intent for intent in intents}
        # Если тег — это запуск подбора породы
        if predicted_tag_second == "подбор_породы":
            await start_breed_dialog(update, context)
            return
        else:
            if predicted_tag_second in tag_to_intent:
                intent = tag_to_intent[predicted_tag_second]
                response = random.choice(intent["responses"])
                await update.message.reply_text(response)
                return
        
    # Если ничего не подошло
    await update.message.reply_text("Не понял вопрос. Попробуйте иначе.")

def main():
    TOKEN = "8000634178:AAHEWtG8rw_V0wf5kc6nELl-1M-VGpYTp28"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
