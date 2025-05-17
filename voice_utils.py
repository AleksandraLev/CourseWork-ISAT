import speech_recognition as sr
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from gtts import gTTS
import io
import uuid

def recognize_voice_from_file(wav_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="ru-RU")
        except sr.UnknownValueError:
            return None

async def send_text_with_voice_button(update, context, text):
    # Генерируем уникальный ключ
    key = str(uuid.uuid4())
    # Сохраняем текст в user_data под этим ключом
    context.user_data[key] = text
    # Передаем в callback_data только ключ
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔊 Послушать ответ", callback_data=f"tts:{key}")
    ]])
    await update.message.reply_text(text, reply_markup=keyboard)

async def callback_tts_handler(update, context):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    if data.startswith("tts:"):
        key = data[4:]
        text_to_speak = context.user_data.get(key)
        if not text_to_speak:
            await query.message.reply_text("Текст для озвучки не найден.")
            return

        tts = gTTS(text=text_to_speak, lang="ru")
        audio = io.BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        
        await context.bot.send_voice(chat_id=query.message.chat.id, voice=audio)