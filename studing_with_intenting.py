import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Загружаем датасет
with open("questions_about_breed.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Подготовка данных
keys = []
questions = []
for question in data["questions_about_breed"]:
    questions.append(question["question"])
    keys.append(question["key"])

# Векторизация текста
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)
y = np.array(keys)

# Обучение модели
model = MultinomialNB()
model.fit(X, y)

# Сохраняем модель и векторизатор
with open("chatbot_model_for_breed_selector.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer_for_breed_selector.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Модель обучена и сохранена!")
