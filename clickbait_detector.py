#!/usr/bin/env python

# imports
import numpy as np

from features import *
from helpers import *


def main():
    # assigns data to variables
    non_clickbait = "non_clickbait_data.txt"
    clickbait = "clickbait_data.txt"

    non_clickbait_headlines = get_data(non_clickbait)  # list
    clickbait_headlines = get_data(clickbait)  # list
    all_headlines = non_clickbait_headlines + clickbait_headlines

    # creates a list of corresponding labels
    # a headline is either a clickbait or not
    # if clickbait it's labeled with 1, otherwise with 0
    non_cb_labels = [0] * len(non_clickbait_headlines)
    cb_labels = [1] * len(clickbait_headlines)
    all_labels = non_cb_labels + cb_labels

    # function words fearures
    stop_words_features = stop_words(all_headlines)

    # syntax features
    pos_tag_features = pos_tags(all_headlines)

    # lexical features
    lexical_features = lexical(all_headlines)

    # punctuation features
    punctuation_features = interpunction(all_headlines)

    # complexity features
    avg_char_feature = avg_char_num(all_headlines)
    avg_char_feature = avg_char_feature.reshape(31998, 1)
    ttr_feature_normalized = ttr_normalized(all_headlines)
    ttr_feature_normalized = ttr_feature_normalized.reshape(31998, 1)
    ttr_feature_raw = ttr_raw(all_headlines)
    ttr_feature_raw = ttr_feature_raw.reshape(31998, 1)
    num_words_feature = num_words(all_headlines)
    num_words_feature = num_words_feature.reshape(31998, 1)
    long_words_feature = long_words(all_headlines)
    long_words_feature = long_words_feature.reshape(31998, 1)
    complexity_features_normalized = np.concatenate(
        (
            avg_char_feature,
            ttr_feature_normalized,
            num_words_feature,
            long_words_feature,
        ),
        axis=1,
    )
    complexity_features_raw = np.concatenate(
        (avg_char_feature, ttr_feature_raw, num_words_feature, long_words_feature),
        axis=1,
    )

    # interrogative features
    question_words_count_feature = q_words_counts(all_headlines)
    questions_feature = questions(all_headlines)
    questions_feature = questions_feature.reshape(31998, 1)

    # all features combined
    all_features = np.concatenate(
        (
            stop_words_features,
            pos_tag_features,
            lexical_features,
            punctuation_features,
            complexity_features_normalized,
            question_words_count_feature,
        ),
        axis=1,
    )

    print(f"Function words:\t{score_m(stop_words_features, all_labels)}")
    print(f"Syntax:\t{score_m(pos_tag_features, all_labels)}")
    print(f"Lexical:\t{score_m(lexical_features, all_labels)}")
    print(f"Punctuation:\t{score_m(punctuation_features, all_labels)}")
    print(
        f"Complexity(normalized):\t{score_m(complexity_features_normalized, all_labels)}"
    )
    print(f"Complexity(raw):\t{score_m(complexity_features_raw, all_labels)}")
    print(f"Question words:\t{score_m(question_words_count_feature, all_labels)}")
    print(f"Questions:\t{score_b(questions_feature, all_labels)}")
    print(f"All features combined:\t{score_m(all_features, all_labels)}")


if __name__ == "__main__":
    main()
