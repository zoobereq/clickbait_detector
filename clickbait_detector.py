#!/usr/bin/env python

# imports
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import string
import re
import urllib.request
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from nltk.translate.ibm_model import Counts


# stopwords, interrogative words, and punctuation
stopwords = set(stopwords.words('english')) 
punctuation = set(string.punctuation + '’') 
q_words = ['what', 'which', 'when', 'where', 'who', 'whom', 'whose', 'why', 'whether', 'how']

# assigns data to variables
non_clickbait_url = "http://www.cs.columbia.edu/~sarahita/CL/non_clickbait_data.txt"
clickbait_url = "http://www.cs.columbia.edu/~sarahita/CL/clickbait_data.txt"

# reads url .txt file into string "data"
def get_data(url):
  data = urllib.request.urlopen(url).read().decode('utf-8')
  return data


def stop_words(headlines):
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


def pos_tags(headlines):
    """Extracts bag-of-words features: POS tags"""
    bow = []
    common_tags = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']
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


def top_30_words(headlines):
    """Isolates the top 30 most common unigrams in corpus"""
    all_words = []
    top_30 = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [''.join(item for item in token if item not in punctuation) for token in tokenized]
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


def lexical(headlines):
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


def interpunction(headlines):
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


def avg_char_num(headlines):
    """Extracts bag-of-words features: 
    average number of characters per word"""
    bow = []
    for headline in headlines:
        token_lengths = []
        tokenized = nltk.word_tokenize(headline.lower())
        for token in tokenized:
            token_lengths.append(len(token))
        headline_avg = sum(token_lengths) / len(token_lengths) # unrounded
        bow.append(headline_avg)
    bow_np = np.array(bow).astype(float)
    return bow_np


def ttr_normalized(headlines):
    """Extracts bag-of-words features: 
    type-to-token ratio for normalized headlines"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [''.join(item for item in token if item not in punctuation) for token in tokenized]
        no_punctuation = [token for token in no_punctuation if token]
        no_stopwords = [token for token in no_punctuation if not token in stopwords]
        num_tokens = len(no_stopwords)
        unique_tokens = set(no_stopwords)
        num_unique_tokens = len(unique_tokens)
        ttr = round((num_unique_tokens / num_tokens), 3)
        bow.append(ttr)
    bow_np = np.array(bow).astype(float)
    return bow_np 


def ttr_raw(headlines):
    """Extracts bag-of-words features: 
    type-to-token ratio for not normalized headlines
    (removed punctuation)"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [''.join(item for item in token if item not in punctuation) for token in tokenized]
        no_punctuation = [token for token in no_punctuation if token]
        num_tokens = len(no_punctuation)
        unique_tokens = set(no_punctuation)
        num_unique_tokens = len(unique_tokens)
        ttr = round((num_unique_tokens / num_tokens), 3)
        bow.append(ttr)
    bow_np = np.array(bow).astype(float)
    return bow_np


def num_words(headlines):
    """Extracts bag-of-words features: 
    number of words per headline
    (removed punctuation)"""
    bow = []
    for headline in headlines:
        tokenized = nltk.word_tokenize(headline.lower())
        no_punctuation = [''.join(item for item in token if item not in punctuation) for token in tokenized]
        no_punctuation = [token for token in no_punctuation if token]
        num_tokens = len(no_punctuation)
        bow.append(num_tokens)
    bow_np = np.array(bow).astype(float)
    return bow_np 


def long_words(headlines):
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


def q_words_counts(headlines):
    """Extracts bag-of-words features: 
    counts of interrogative words per headline"""
    bow = []
    for headline in headlines:
        q_word_counts = []
        tokenized = nltk.word_tokenize(headline.lower())
        for q_word in q_words:
            q_count = tokenized.count(q_word)
            q_word_counts.append(q_count)
        bow.append(q_word_counts)
    bow_np = np.array(bow).astype(float)
    return bow_np


def questions(headlines):
    """Extracts bag-of-words features: 
    Checks if a headline begins with an interrogative
    i.e. if the headline is a question"""
    bow = []
    for headline in headlines:
        is_q = 0
        tokenized = nltk.word_tokenize(headline.lower())
        for q_word in q_words:
            if tokenized[0] == q_word:
                is_q = 1
            else:
                is_q = 0
        bow.append(is_q)
    bow_np = np.array(bow).astype(float)  
    return bow_np


def score_m(data, target):
    """runs the Multinomial Naive Bayes classifier with 10-fold cross validation
    and reports mean accuracy """
    X = data
    Y = np.array(target)
    score = cross_val_score(MultinomialNB(), X, Y, scoring='accuracy', cv=10)
    return round(score.mean(), 3)  

def score_b(data, target):
    """runs the Bernoulli Naive Bayes classifier with 10-fold cross validation
    and reports mean accuracy """
    X = data
    Y = np.array(target)
    score = cross_val_score(BernoulliNB(), X, Y, scoring='accuracy', cv=10)
    return round(score.mean(), 3)     


def main():
    non_clickbait_data = get_data(non_clickbait_url) # string
    clickbait_data = get_data(clickbait_url) # string

    # combines clickbait and non-clickbait data in a single list
    non_clickbait_headlines = non_clickbait_data.rstrip('\n').split('\n') # list1
    clickbait_headlines = clickbait_data.rstrip('\n').split('\n') # list2
    all_headlines = non_clickbait_headlines + clickbait_headlines

    # create a list of corresponding labels
    # a headline is either a clickbait or not
    # if clickbait it's labeled with 1, otherwise with 0
    non_cb_labels = [0] * len(non_clickbait_headlines)
    cb_labels = [1] * len(clickbait_headlines)
    all_labels = non_cb_labels + cb_labels

    # function words
    stop_words_features = stop_words(all_headlines)

    # syntax
    pos_tag_features = pos_tags(all_headlines)

    # lexical
    lexical_features = lexical(all_headlines)

    # punctuation
    punctuation_features = interpunction(all_headlines)

    # complexity
    avg_char_feature = avg_char_num(all_headlines)
    avg_char_feature = avg_char_feature.reshape(31998,1)
    ttr_feature_normalized = ttr_normalized(all_headlines)
    ttr_feature_normalized = ttr_feature_normalized.reshape(31998,1)
    ttr_feature_raw = ttr_raw(all_headlines)
    ttr_feature_raw = ttr_feature_raw.reshape(31998,1)
    num_words_feature = num_words(all_headlines)
    num_words_feature = num_words_feature.reshape(31998,1)
    long_words_feature = long_words(all_headlines)
    long_words_feature = long_words_feature.reshape(31998,1)
    complexity_features_normalized = np.concatenate((avg_char_feature, ttr_feature_normalized, num_words_feature, long_words_feature), axis = 1)
    complexity_features_raw = np.concatenate((avg_char_feature, ttr_feature_raw, num_words_feature, long_words_feature), axis = 1)
    
    # my feature sets
    question_words_count_feature = q_words_counts(all_headlines)
    questions_feature = questions(all_headlines)
    questions_feature = questions_feature.reshape(31998,1)

    # all features combined
    all_features = np.concatenate((stop_words_features, pos_tag_features, lexical_features, punctuation_features, complexity_features_normalized, question_words_count_feature), axis = 1)

    print(f"Function words:\t{score_m(stop_words_features, all_labels)}")
    print(f"Syntax:\t{score_m(pos_tag_features, all_labels)}")
    print(f"Lexical:\t{score_m(lexical_features, all_labels)}")
    print(f"Punctuation:\t{score_m(punctuation_features, all_labels)}")
    print(f"Complexity(normalized):\t{score_m(complexity_features_normalized, all_labels)}")
    print(f"Complexity(raw):\t{score_m(complexity_features_raw, all_labels)}")
    print(f"Question words:\t{score_m(question_words_count_feature, all_labels)}")
    print(f"Questions:\t{score_b(questions_feature, all_labels)}")
    print(f"All features combined:\t{score_m(all_features, all_labels)}")      
    

if __name__ == "__main__":
    main()