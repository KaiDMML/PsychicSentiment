import os
import numpy as np

from flask import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold



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


# Create a corpus with each document having one string
root_dir = 'txt_sentoken'
corpus = make_Corpus(root_dir)
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
print pos_avg, neg_avg, pos_cnt, neg_cnt, lenn
thres = 0.6
print len(word_polarity)

for i in range(len(corpus)):
    new_sent = []
    for word in corpus[i].split(' '):
        word=word.lower()

        if word in word_polarity:
            if word_polarity[word]['pos'] != 0 and word_polarity[word]['pos'] > thres:
                continue
            if word_polarity[word]['neg'] != 0 and word_polarity[word]['neg'] > thres:
                continue
        new_sent.append(word)
    corpus[i]=' '.join(new_sent)

# # Stratified 10-cross fold validation with SVM and Multinomial NB
labels = np.zeros(2000)
labels[0:1000] = 0
labels[1000:2000] = 1

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
    print 'something'

print totalMatSvm
print totalsvm / 2000.0
print totalMatNB
print totalNB / 2000.0

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
train_corpus_tf_idf = vectorizer.fit_transform(corpus)

model1 = LinearSVC()
model1.fit(train_corpus_tf_idf, labels)

psyc_data={}
with open('newsdata.json') as data_file:
    psyc_data=json.load(data_file)

all_news=[]
for news in psyc_data:
    if news:
        all_news.append(news['article_text'])



for i in range(len(all_news)):
    new_sent = []
    for word in all_news[i].split(' '):
        word=word.lower()

        if word in word_polarity:
            if word_polarity[word]['pos'] != 0 and word_polarity[word]['pos'] > thres:
                continue
            if word_polarity[word]['neg'] != 0 and word_polarity[word]['neg'] > thres:
                continue
        new_sent.append(word)
        all_news[i]=' '.join(new_sent)


result1 = model1.predict(vectorizer.transform(all_news))


for i in range(len(all_news)):
    print all_news[i]
    print result1[i]

