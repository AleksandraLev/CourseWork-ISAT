import speech_recognition as sr
import pyttsx3

def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Говорите...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")
            print("Вы сказали:", text)
            return text
        except:
            return "Извините, не расслышал."

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

# Пример:
# speak_text("Здравствуйте! Чем могу помочь?")
