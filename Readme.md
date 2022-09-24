## Clickbait Detector

### Motivation
There is an overwhelming amount of news information available online.  Some news headlines are known as clickbait – they aim to attract users to click on a link but the articles that they link to may not be of value or interest to the reader.  This program automatically distinguishes between clickbait and non-clickbait headlines.

### Data
Two corpora of clickbait and non-clicbait headlines are included.  Each corpus counts 16,000 headlines, for a total of 32,000 headlines.

### Code
The code is informed by [the paper] (https://arxiv.org/pdf/1610.09786.pdf) by Chakraborty et al. (2016).  The program loads the data, extracts sets of features as frequency-count vectors, and uses them to train a Naive Bayes classifier.  The classifier accuracy is generated using 10-fold cross-validation and output for each feature set individually.  The program extracts the following features:
- **Stop words:** counts for each function word (from the NLTK stopwords list)
- **Syntactic:**  counts for the following 10 common POS tags: `['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']`
- **Lexical:** counts for 30 most common unigrams in the entire corpus
- **Punctuation:**  Counts for each punctuation mark in `string.punctuation`
- **Complexity:** 
    - average number of characters per word
    - Type-to-token ratio (the number of unique words / the total number of words)
    - Count of *long* words - words with at least 6 letters
- **Interrogative words:** counts of the common English interrogatives

### Accuracy
*Clickbait Detector* achieves 91% accuracy for all features combined, meaning that it correctly identifies over 9 out of 10 headlines.
