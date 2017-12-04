import time
import stomp
import logging
logging.basicConfig(level=logging.DEBUG)
import json
import argparse
import pickle
import json,sys
from tqdm import tqdm
reload(sys)
import string
import os
import copy
import string

import numpy as np
import sys
from flask import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
sys.setdefaultencoding('utf-8')

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
def remove_high_sentiment_words(corpus, threshold1,threshold2,word_polarity):
    for i in range(len(corpus)):
        new_sent = []
        for word in corpus[i].split(' '):
            word = word.lower()

            if word in word_polarity:
                if word_polarity[word]['pos'] != 0 and word_polarity[word]['pos'] > threshold1:
                    continue
                if word_polarity[word]['neg'] != 0 and word_polarity[word]['neg'] > threshold2:
                    continue
            new_sent.append(word)
        corpus[i]=' '.join(new_sent)
        corpus[i] = ''.join([x for x in corpus[i] if x in string.ascii_letters + '\'- '])
        #print corpus[i]
    return corpus

# This function is used to train the data
def train_SVM_model(corpus, labels):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
    train_corpus_tf_idf = vectorizer.fit_transform(corpus)

    model1 = LinearSVC()
    model1.fit(train_corpus_tf_idf, labels)
    return model1,vectorizer

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

postiveTrainThreshold = 2
negativeTrainThreshold = 2

postiveTestThreshold = 0.35
negativeTestThreshold = 0.4

corpus = make_Corpus('txt_sentoken')
word_polarity = get_polarity_words()
labels = np.zeros(2000)
labels[0:1000] = 0
labels[1000:2000] = 1
corpus2 = remove_high_sentiment_words(copy.copy(corpus), postiveTrainThreshold, negativeTrainThreshold, word_polarity)
model1, vectorizer = train_SVM_model(corpus2, labels)

URL = "psychic.cra.com"
PORT = 61613  # stomp.

conn2 = stomp.Connection([(URL, PORT)])
# Disable content-length header to send as TextMessage.
conn2.auto_content_length = False
conn2.start()
conn2.connect('authtest', 'authtest', wait=True)
conn2.subscribe(destination='/topic/ADD_DATA', ack='auto', id=1)

class MyListener(stomp.ConnectionListener):
    def on_error(self, headers, message):
        print('received an error "%s"' % message)

    def on_message(self, headers, message):
        print('received a message "%s"' % message[:100])
        mentions = json.loads(message)
        all_inp2 = remove_high_sentiment_words(copy.copy([message['body']]), postiveTestThreshold, negativeTestThreshold,
                                               word_polarity)

        # Predict psychic data sentiment
        result2 = model1.predict(vectorizer.transform(all_inp2))


        for r in range(len(result2)):
            mentions['hierar_sentiment'] = r

        conn2.send(body=json.dumps(mentions), destination='/topic/HIERAR_SENTIMENT')


def main():
    global args, sentiment_output_processor, conn

    URL = "psychic.cra.com"
    PORT = 61613
    conn = stomp.Connection(host_and_ports=[(URL, PORT)],
                            prefer_localhost=False,
                            auto_content_length=False)

    listener = MyListener()
    conn.set_listener("", listener)
    conn.start()
    conn.connect('authtest', 'authtest', wait=True)
    conn.subscribe(destination='/topic/RESULT_AVAILABLE', id=1, ack='auto')
    # print ('[+] Connection started')

    # print "Listening on " + args.psydata_server_host + ":" + str(args.psydata_port)
    while conn.is_connected():
        time.sleep(5)


if __name__ == "__main__":
    main()