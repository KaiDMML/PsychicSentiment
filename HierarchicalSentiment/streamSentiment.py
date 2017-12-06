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
    return corpus

# Load model
model1,vectorizer,word_polarity=pickle.load(open('hierar_sentiment.model', 'rb'))

# Set Test Threshold
postiveTestThreshold = 0.35
negativeTestThreshold = 0.4

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