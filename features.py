#!/usr/bin/env python

# imports
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
import string
from collections import Counter

import numpy as np
from nltk.corpus import stopwords
from numpy import ndarray

# stopwords, interrogative words, and punctuation
stopwords = set(stopwords.words("english"))
punctuation = set(string.punctuation + "’")
q_words = [
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "whether",
    "how",
    "did",
    "does",
    "have",
    "has",
    "had",
    "is",
    "was",
    "were",
]
interrogatives = [
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "whether",
    "how",
]


def stop_words(headlines: list) -> ndarray:
    """Extracts bag-of-words features: stop words"""
    bow = []
    for headline in headlines:
        counts = []
        tokens = nltk.word_tokenize(headline.lower())
        for stopword in stopwords:
            stopword_count = tokens.count(stopword)
            counts.append(stopword_count)
        bow.append(counts)
    bow_np = np.array(bow).astype(float)
    return bow_np


def pos_tags(headlines: list) -> ndarray:
    """Extracts bag-of-words features: POS tags"""
    bow = []
    common_tags = ["NN", "NNP", "DT", "IN", "JJ", "NNS", "CC", "PRP", "VB", "VBG"]
    for headline in headlines:
        headline_tags = []
        headline_tag_counts = []
        tokens = nltk.word_tokenize(headline.lower())
        tagged_tokens = nltk.pos_tag(tokens)
        for tagged_token in tagged_tokens:
            tag = tagged_token[1]
            headline_tags.append(tag)
        for tag in common_tags:
            tag_count = headline_tags.count(tag)
            headline_tag_counts.append(tag_count)
        bow.append(headline_tag_counts)
    bow_np = np.array(bow).astype(float)
    return bow_np


def top_30_words(headlines: list) -> list:
    """Isolates the top 30 most common unigrams in corpus"""
    all_words = []
    top_30 = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [
            "".join(item for item in token if item not in punctuation)
            for token in tokenized
        ]
        no_punctuation = [token for token in no_punctuation if token]
        no_stopwords = [token for token in no_punctuation if not token in stopwords]
        for token in no_stopwords:
            all_words.append(token)
    token_counts = Counter(all_words)
    top_word_counts = token_counts.most_common(30)
    for item in top_word_counts:
        top_word = item[0]
        top_30.append(top_word)
    return top_30


def lexical(headlines: list) -> ndarray:
    """Extracts bag-of-words features: lexical"""
    bow = []
    top_30 = top_30_words(headlines)
    for headline in headlines:
        counts = []
        tokenized = nltk.word_tokenize(headline.lower())
        for word in top_30:
            top_word_count = tokenized.count(word)
            counts.append(top_word_count)
        bow.append(counts)
    bow_np = np.array(bow).astype(float)
    return bow_np


def interpunction(headlines: list) -> ndarray:
    """Extracts bag-of-words features: punctuation"""
    bow = []
    for headline in headlines:
        counts = []
        tokenized = nltk.word_tokenize(headline.lower())
        for item in punctuation:
            punctuation_count = tokenized.count(item)
            counts.append(punctuation_count)
        bow.append(counts)
    bow_np = np.array(bow).astype(float)
    return bow_np


def avg_char_num(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    average number of characters per word"""
    bow = []
    for headline in headlines:
        token_lengths = []
        tokenized = nltk.word_tokenize(headline.lower())
        for token in tokenized:
            token_lengths.append(len(token))
        headline_avg = sum(token_lengths) / len(token_lengths)  # unrounded
        bow.append(headline_avg)
    bow_np = np.array(bow).astype(float)
    return bow_np


def ttr_normalized(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    type-to-token ratio for normalized headlines"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [
            "".join(item for item in token if item not in punctuation)
            for token in tokenized
        ]
        no_punctuation = [token for token in no_punctuation if token]
        no_stopwords = [token for token in no_punctuation if not token in stopwords]
        num_tokens = len(no_stopwords)
        unique_tokens = set(no_stopwords)
        num_unique_tokens = len(unique_tokens)
        ttr = round((num_unique_tokens / num_tokens), 3)
        bow.append(ttr)
    bow_np = np.array(bow).astype(float)
    return bow_np


def ttr_raw(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    type-to-token ratio for not normalized headlines
    (removed punctuation)"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [
            "".join(item for item in token if item not in punctuation)
            for token in tokenized
        ]
        no_punctuation = [token for token in no_punctuation if token]
        num_tokens = len(no_punctuation)
        unique_tokens = set(no_punctuation)
        num_unique_tokens = len(unique_tokens)
        ttr = round((num_unique_tokens / num_tokens), 3)
        bow.append(ttr)
    bow_np = np.array(bow).astype(float)
    return bow_np


def num_words(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    number of words per headline
    (removed punctuation)"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [
            "".join(item for item in token if item not in punctuation)
            for token in tokenized
        ]
        no_punctuation = [token for token in no_punctuation if token]
        num_tokens = len(no_punctuation)
        bow.append(num_tokens)
    bow_np = np.array(bow).astype(float)
    return bow_np


def long_words(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    number of words with at least 6 characters
    per headline"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        long_words_headline = []
        for token in tokenized:
            if len(token) >= 6:
                long_words_headline.append(token)
        bow.append(len(long_words_headline))
    bow_np = np.array(bow).astype(float)
    return bow_np


def q_words_counts(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    counts of interrogative words per headline"""
    bow = []
    for headline in headlines:
        q_word_counts = []
        tokenized = nltk.word_tokenize(headline.lower())
        for q_word in interrogatives:
            q_count = tokenized.count(q_word)
            q_word_counts.append(q_count)
        bow.append(q_word_counts)
    bow_np = np.array(bow).astype(float)
    return bow_np


def questions(headlines: list) -> ndarray:
    """Extracts bag-of-words features: 
    Checks if a headline begins with an interrogative
    i.e. if the headline is a question"""
    bow = []
    for headline in headlines:
        is_q = 0
        tokenized = nltk.word_tokenize(headline.lower())
        for q_word in q_words:
            if tokenized[0] == q_word and tokenized[-1] == "?":
                is_q = 1
            else:
                is_q = 0
        bow.append(is_q)
    bow_np = np.array(bow).astype(float)
    return bow_np
