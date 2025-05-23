import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import CountVectorizer

# Загрузка данных
with open("questions_about_breed.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Подготовка данных
answers = []
keys = []
for question in data["questions_about_breed"]:
    for option in question["options"]:
        answers.append(option)
        keys.append(question["key"])

# Векторизация текста
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(answers)
y = np.array(keys)

# Модель
model = MultinomialNB()

# Кривая обучения
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=2, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Средние значения точности
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

# Визуализация
plt.figure(figsize=(10, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_scores_mean, label='Обучающая выборка')
plt.plot(train_sizes, test_scores_mean, label='Проверочная выборка')
plt.title('Кривая обучения (Точность)')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Точность')
plt.legend()

# График ошибки
plt.subplot(1, 2, 2)
plt.plot(train_sizes, 1 - train_scores_mean, label='Обучающая выборка')
plt.plot(train_sizes, 1 - test_scores_mean, label='Проверочная выборка')
plt.title('Кривая обучения (Ошибка)')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Ошибка (1 - точность)')
plt.legend()

plt.tight_layout()
plt.show()
