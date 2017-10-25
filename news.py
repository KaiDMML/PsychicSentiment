import os
import numpy as np
import sys
from flask import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


# This function is used to read the data from txt_sentoken which ha 1000 positive and 1000 negative sentiment data.
def make_Corpus(root_dir):
    polarity_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    corpus = []
    for polarity_dir in polarity_dirs:
        reviews = [os.path.join(polarity_dir, f) for f in os.listdir(polarity_dir)]
        for review in reviews:
            doc_string = ""
            with open(review) as rev:
                for line in rev:
                    doc_string = doc_string + line
            if not corpus:
                corpus = [doc_string]
            else:
                corpus.append(doc_string)
    return corpus


# This function is used to read SentiWordNet.txt file which has more than 89k words with sentiment score.
def get_polarity_words():
    word_polarity = {}
    pos_cnt = 0
    neg_cnt = 0
    pos_avg = 0
    neg_avg = 0
    lenn = 0

    with open('SentiWordNet.txt') as data_file:
        for line in data_file:
            if (line[0] == '#'):
                continue
            sp = line.split('\t')
            sp[4] = sp[4].split('#')[0]
            if sp[4] not in word_polarity:
                lenn += 1
                if not sp[2] and not sp[3]:
                    print sp[2], sp[3]
                    continue
                if float(sp[2]) != 0:
                    pos_cnt = pos_cnt + 1
                    pos_avg = pos_avg + float(sp[2])
                if float(sp[3]) != 0:
                    neg_cnt = neg_cnt + 1
                    neg_avg = neg_avg + float(sp[3])
                word_polarity[sp[4].lower()] = {'pos': float(sp[2]), 'neg': float(sp[3])}

    pos_avg = pos_avg / pos_cnt
    neg_avg = neg_avg / neg_cnt
    print "Average Positive Sentiment",pos_avg
    print "Average Negative Sentiment",neg_avg
    print "Total Words in SentiWordNet",len(word_polarity)

    return word_polarity


# This function is used to remove words with high polarity
def remove_high_sentiment_words(corpus, threshold,word_polarity):
    for i in range(len(corpus)):
        new_sent = []
        for word in corpus[i].split(' '):
            word = word.lower()

            if word in word_polarity:
                if word_polarity[word]['pos'] != 0 and word_polarity[word]['pos'] > threshold:
                    continue
                if word_polarity[word]['neg'] != 0 and word_polarity[word]['neg'] > threshold:
                    continue
            new_sent.append(word)
        corpus[i] = ' '.join(new_sent)
    return corpus


# This function is used to perform K fold cross validation on the training dataset
def k_fold_vaildation(corpus, labels):
    kf = StratifiedKFold(n_splits=10)

    totalsvm = 0  # Accuracy measure on 2000 files
    totalNB = 0
    totalMatSvm = np.zeros((2, 2))  # Confusion matrix on 2000 files
    totalMatNB = np.zeros((2, 2))

    for train_index, test_index in kf.split(corpus, labels):
        X_train = [corpus[i] for i in train_index]
        X_test = [corpus[i] for i in test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
        train_corpus_tf_idf = vectorizer.fit_transform(X_train)
        test_corpus_tf_idf = vectorizer.transform(X_test)

        model1 = LinearSVC()
        model2 = MultinomialNB()
        model1.fit(train_corpus_tf_idf, y_train)
        model2.fit(train_corpus_tf_idf, y_train)
        result1 = model1.predict(test_corpus_tf_idf)
        result2 = model2.predict(test_corpus_tf_idf)

        totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1)
        totalMatNB = totalMatNB + confusion_matrix(y_test, result2)
        totalsvm = totalsvm + sum(y_test == result1)
        totalNB = totalNB + sum(y_test == result2)

    print "SVM Confusion Matrix"
    print totalMatSvm

    print "SVM Accuracy"
    print totalsvm / 2000.0

    print "NB Confusion Matrix"
    print totalMatNB

    print "NB Accuracy"
    print totalNB / 2000.0


# This function is used to train the data
def train_SVM_model(corpus, labels):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
    train_corpus_tf_idf = vectorizer.fit_transform(corpus)

    model1 = LinearSVC()
    model1.fit(train_corpus_tf_idf, labels)
    return model1,vectorizer


# This function is used to get the psychic dataset
def get_psychic_data():
    psyc_data = {}
    with open('newsdata.json') as data_file:
        psyc_data = json.load(data_file)

    all_news = []
    for news in psyc_data:
        if news:
            all_news.append(news['article_text'])
    return all_news


def main(argv):
    # Create a corpus with each document having one string
    corpus = make_Corpus('txt_sentoken')

    # Get words with sentiment polarity
    word_polarity = get_polarity_words()

    # Word Sentiment threshold
    threshold = 0.6

    corpus = remove_high_sentiment_words(corpus, threshold,word_polarity)

    # Stratified 10-cross fold validation with SVM and Multinomial NB
    labels = np.zeros(2000)
    labels[0:1000] = 0
    labels[1000:2000] = 1

    k_fold_vaildation(corpus, labels)

    # Train SVM on whole data
    model1,vectorizer = train_SVM_model(corpus, labels)

    # Load Psychic data and remove high sentiment words
    all_news = get_psychic_data()
    all_news = remove_high_sentiment_words(all_news, threshold,word_polarity)

    # Predict psychic data sentiment
    result1 = model1.predict(vectorizer.transform(all_news))

    # Print the predicted sentiment data
    for i in range(len(all_news)):
        print all_news[i]
        print result1[i]


if __name__=='__main__':
    main(sys.argv)

