import re
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser
from difflib import get_close_matches
from rapidfuzz import process


# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
embedding = NewsEmbedding()
morph_tagger = NewsMorphTagger(embedding)
syntax_parser = NewsSyntaxParser(embedding)

def clean_text(text):
    """Удаление лишних символов, пробелов, приведение к нижнему регистру"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Удаление пунктуации
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize(text):
    """Приведение слов к нормальной форме"""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    # Лемматизация всех слов
    lemmatized_words = [_.lemma or _.text for _ in doc.tokens]
    print(f"Lemmatized: {lemmatized_words}")
    return ' '.join(lemmatized_words)


def correct_spelling_tag(user_input, pattern_to_tag, cutoff=0.6):
    """
    Исправление опечаток с использованием расстояния Левенштейна.
    Ищем наиболее похожий вариант из valid_options.
    """
    print(f"Исходный текст: {user_input}")

    # Лемматизируем ответ пользователя
    #lemmatized_input = lemmatize(user_input)

    # Находим наиболее похожие допустимые ответы
    patterns = list(pattern_to_tag.keys())
    matches = get_close_matches(user_input, patterns, n=1, cutoff=cutoff)

    if matches:
        matched_pattern = matches[0]
        matched_tag = pattern_to_tag[matched_pattern]
        print(f"Найден похожий шаблон: '{matched_pattern}' → тег '{matched_tag}'")
        return matched_tag, True
    else:
        # Если не найдено, возвращаем лемматизированный текст
        return user_input, False

def correct_spelling_words(user_input, pattern_to_tag, cutoff=0.6):
    """
    Исправление опечаток с использованием расстояния Левенштейна.
    Ищем наиболее похожий вариант из valid_options.
    """
    print(f"Исходный текст: {user_input}")

    # Лемматизируем ответ пользователя
    #lemmatized_input = lemmatize(user_input)

    # Находим наиболее похожие допустимые ответы
    patterns = list(pattern_to_tag.keys())
    matches = get_close_matches(user_input, patterns, n=1, cutoff=cutoff)

    if matches:
        matched_pattern = matches[0]
        matched_tag = pattern_to_tag[matched_pattern]
        print(f"Найден похожий шаблон: '{matched_pattern}' → тег '{matched_tag}'")
        return matched_pattern
    else:
        # Если не найдено, возвращаем лемматизированный текст
        return user_input
