import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Загружаем датасет
with open("answers_for_advertisement.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Подготовка данных
tags = []
patterns = []
for intent  in data["answers_for_advertisement"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Векторизация текста
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = np.array(tags)

# Обучение модели
model = MultinomialNB()
model.fit(X, y)

# Сохраняем модель и векторизатор
with open("chatbot_model_for_advertisement.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer_for_advertisement.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Модель обучена и сохранена!")
