import json,sys
from pymongo import MongoClient
from tqdm import tqdm
from datetime import datetime
import lib.twokenize
import nltk.corpus
import numpy
import sklearn.preprocessing
import sklearn.feature_extraction.text
from scipy.sparse import csr_matrix as to_sparse
import scipy.sparse
import sklearn.linear_model
import pickle
import json,sys
from pymongo import MongoClient
from tqdm import tqdm
from bson import ObjectId
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
    #print train_corpus_tf_idf
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


def put_heirarchical_forum():
    postiveTrainThreshold = 2
    negativeTrainThreshold = 2

    postiveTestThreshold = 0.35
    negativeTestThreshold = 0.4

    #f_result = open('historical_twitter_forum_examplet1.json', 'w+')
    corpus = make_Corpus('txt_sentoken')
    word_polarity = get_polarity_words()
    labels = np.zeros(2000)
    labels[0:1000] = 0
    labels[1000:2000] = 1
    corpus2 = remove_high_sentiment_words(copy.copy(corpus), postiveTrainThreshold, negativeTrainThreshold, word_polarity)
    model1, vectorizer = train_SVM_model(corpus2, labels)
    all_inp = []
    d={}


    # with open('historical_twitter_forum_example.json', 'r') as json_data:
    #     d = json.load(json_data)
    #     for k in d.keys():
    #         for e in d[k]:
    #             for inp in e['_source']['data']['posts']:
    #                 if inp['post']['content'][0]:
    #                     all_inp.append(inp['post']['content'][0])

    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(
        '/home/local/ASUAD/sshar107/Downloads/20171207-125044/newsscraper/uploadjson/json_accent_events') if isfile(
        join('/home/local/ASUAD/sshar107/Downloads/20171207-125044/newsscraper/uploadjson/json_accent_events', f))]



    for file in onlyfiles:
        with open('/home/local/ASUAD/sshar107/Downloads/20171207-125044/newsscraper/uploadjson/json_accent_events/'+file, 'r') as json_data:
            d = json.load(json_data)
            try:
                d=d['data']
                for dd in d:
                    print dd
                    dd=dd['supporting_snippets']
                    for k in dd:
                        print k
                        all_inp.append(k)
            except:
                pass
    print 'sho',all_inp
    all_inp2 = remove_high_sentiment_words(copy.copy(all_inp), postiveTestThreshold, negativeTestThreshold, word_polarity)

    # Predict psychic data sentiment
    result2 = model1.predict(vectorizer.transform(all_inp2))
    print result2
    indr=0
    pos2=0
    neg2=0
    # for k in d.keys():
    #     for e in d[k]:
    #         for inp in e['_source']['data']['posts']:
    #             if inp['post']['content'][0]:
    #                 inp['post']['sentiment2'] = result2[indr]
    #                 if int(inp['post']['sentiment2']) == 1:
    #                     pos2+=1
    #                 else:
    #                     neg2+=1
    #                 indr+=1


    print pos2,neg2
    #json.dump(d, f_result)


def put_heirarchical(attack):
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
    all_inp=[]
    all=[]
    with open('historical_twitter_'+attack+'.json','r') as f_input:
        for line in tqdm(f_input):
            tt_json = json.loads(line)
            #print tt_json
            all_inp.append(tt_json['body'])
            all.append(tt_json)
    all_inp2 = remove_high_sentiment_words(copy.copy(all_inp), postiveTestThreshold, negativeTestThreshold, word_polarity)
    pos2 = 0
    neg2 = 0
    # Predict psychic data sentiment
    result2 = model1.predict(vectorizer.transform(all_inp2))
    f_result = open('hierar_sentiment_' + attack + '.json', 'w+')
    for r in range(len(result2)):
        all[r]['sentiment']=result2[r]
        if int(result2[r]) == 1:
            pos2 += 1
        else:
            neg2 += 1
        json.dump(all[r], f_result)
        f_result.write('\n')

    print 'Positive Tweets:',pos2,'Negative Tweets:',neg2,'Total Tweet Count:',pos2+neg2


if __name__ == '__main__':

    attack = 'darknetrelaunch_cratagged'

    #put_heirarchical_forum()
    put_heirarchical_forum()