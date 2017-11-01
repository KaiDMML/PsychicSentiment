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

positive_emojis = set([':-)', '(-:', '=)', '(=', '(:', ':)', ':-(', ':D', '^_^', '^__^', '^___^', ':d', 'd:', ': )', '( :', '8)', \
              '(8', '8 )', ';)', '; )', '; )', '( ;', ';-)', '(-;', '(;'])

negative_emojis = set([':-(', ')-:', '=(', ')=', ':(', '):', '8(', ')8'])

emojiList = positive_emojis.union(negative_emojis)

punctuation = set([',', ';', '.', ':', '.', '!', '?', '\"', '*', '\'', '(', ')', '-'])

# TODO These 'adverbs' are from previous work. These could be updated to represent more adverbs
adverbs = set(['very', 'extremly', 'highly'])

# TODO Unused feature extraction, for now

class SentimentClassifier(object):

    def __init__(self, lang="en", ngrams=2):
        self.tokenizer = lib.twokenize.tokenize
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.lang = lang
        self.target_not = 'not' # TODO Handle case of other languages
        self.adverbs = adverbs
        # self.stopwords = self.stopwords.difference(adverbs)

        self.__vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,ngrams), binary=True, tokenizer=self.tokenizer)
        self.__regression_model = sklearn.linear_model.LogisticRegression()

    def train_on_document_list(self, documents):
        corpus, text_features, labels = self.preprocess(documents)
        # Clean as in, only containing documents that we have valid 'labels' for
        clean_corpus, clean_feature_matrix, clean_labels = self.clean(corpus, text_features, labels)
        # Fit vectorizer
        n_gram_features = self.__vectorizer.fit_transform(clean_corpus)
        scaled_sparse_features = to_sparse(sklearn.preprocessing.scale(clean_feature_matrix))
        feature_matrix = scipy.sparse.hstack([n_gram_features, scaled_sparse_features], format='csr')

        # clean_corpus, feature_matrix, clean_labels
        self.__regression_model.fit(feature_matrix, clean_labels)

        return (clean_corpus, feature_matrix, clean_labels)

    def obtain_document_list_probability(self, documents):
        corpus, text_features, labels = self.preprocess(documents)
        n_gram_features = self.__vectorizer.transform(corpus)
        scaled_sparse_features = to_sparse(sklearn.preprocessing.scale(text_features))
        feature_matrix = scipy.sparse.hstack([n_gram_features, scaled_sparse_features], format='csr')
        return self.__regression_model.predict_proba(feature_matrix)

    def obtain_document_list_prediction(self, documents):
        corpus, text_features, labels = self.preprocess(documents)
        n_gram_features = self.__vectorizer.transform(corpus)
        scaled_sparse_features = to_sparse(sklearn.preprocessing.scale(text_features))
        feature_matrix = scipy.sparse.hstack([n_gram_features, scaled_sparse_features], format='csr')
        return self.__regression_model.predict(feature_matrix)

    def demonstration(self, text_list, pred):
        self.train_on_document_list(text_list)
        print self.obtain_document_list_probability(pred)

    def obtain_ngrams(self, text_list, fit_bool):
        if fit_bool:
            n_gram_features = self.__vectorizer.fit_transform(text_list)
        else:
            n_gram_features = self.__vectorizer.transform(text_list)
        return n_gram_features

    def obtain_feature_matrix(self, text_list, fit=False):
        corpus, text_features, labels = self.preprocess(text_list)

        n_gram_features = self.obtain_ngrams(corpus, fit_bool=fit)
        scaled_sparse_features = to_sparse(sklearn.preprocessing.scale(text_features))
        feature_matrix = scipy.sparse.hstack([n_gram_features, scaled_sparse_features], format='csr')


        return (corpus, feature_matrix, labels)

    def clean(self, document_list, feature_matrix, labels):
        valid_labels = set([-1, 1])
        valid_indexes = numpy.array([index for index, label in enumerate(labels) if label in valid_labels])

        cleaned_documents = document_list[valid_indexes]
        cleaned_feature_matrix = feature_matrix[valid_indexes]
        cleaned_labels = labels[valid_indexes]
        return (cleaned_documents, cleaned_feature_matrix, cleaned_labels)

    def preprocess(self, text_list):

        repetition_counts = []
        hashtag_counts = []
        questionmark_counts = []
        exclaimationmark_counts = []
        negation_counts = []
        text_stringlengths = []
        text_labels = []

        for text_index, text in enumerate(text_list):
            tokenized_text = self.tokenizer(text)
            # General, remove stopwords
            nostopwords_text = [word for word in tokenized_text if word not in self.stopwords and not word.startswith('RT')]

            text_repetition_count = 0.
            text_hashtag_count = 0.
            text_exclaimationmark_count = 0.
            text_questionmark_count = 0.
            text_negation_count = 0.

            # +1 for positive
            # -1 for negative
            # INFINITY for none
            # -INFINITY for conflicting
            text_label = numpy.inf

            clean_text = nostopwords_text

            for word_in_text_index, word_in_text in enumerate(clean_text):
                if word_in_text.startswith(('.@', '@')): # Twitter specific mention
                    clean_text[word_in_text_index] = '___mention___'
                elif word_in_text.startswith(('www', 'http')): # Clean urls
                    clean_text[word_in_text_index] = '___url___'
                elif word_in_text.startswith('!'):
                    text_exclaimationmark_count += 1
                elif word_in_text.startswith('?'):
                    text_questionmark_count += 1
                elif word_in_text.startswith('#'):
                    text_hashtag_count += 1
                elif word_in_text == self.target_not:
                    text_negation_count += 1
                    if word_in_text_index + 1 < len(clean_text):
                        clean_text[word_in_text_index] = '' # Remove not
                        clean_text[word_in_text_index+1] = self.__target_not+'__'+clean_text[word_in_text_index+1] # Preappend removed not to next word
                elif word_in_text in positive_emojis:
                    if numpy.isposinf(text_label) or text_label == +1:
                        text_label = +1
                    else:
                        text_label = numpy.NINF
                elif word_in_text in negative_emojis:
                    if numpy.isposinf(text_label) or text_label == -1:
                        text_label = -1
                    else:
                        text_label = numpy.NINF

            repetition_counts.append(text_repetition_count)
            hashtag_counts.append(text_hashtag_count)
            questionmark_counts.append(text_questionmark_count)
            exclaimationmark_counts.append(text_exclaimationmark_count)
            negation_counts.append(text_negation_count)
            text_stringlengths.append(len(clean_text))
            text_labels.append(text_label)

            # Removing punctuation
            nopunctuation_text_array = [''.join([char for char in word if char not in punctuation]) for word in clean_text if len(word)>2]
            nopunctuation_text = ' '.join(nopunctuation_text_array)
            text_list[text_index] = nopunctuation_text

        np_text_list = numpy.array(text_list).transpose()

        np_repetition_counts = numpy.array(repetition_counts)
        np_hashtag_counts = numpy.array(hashtag_counts)
        np_questionmark_counts = numpy.array(questionmark_counts)
        np_exclaimationmark_counts = numpy.array(exclaimationmark_counts)
        np_negation_counts = numpy.array(negation_counts)
        np_textstringlengths = numpy.array(text_stringlengths)
        np_text_labels = numpy.array(text_labels).transpose()

        features = numpy.vstack((np_repetition_counts, np_hashtag_counts, np_questionmark_counts, np_exclaimationmark_counts,
                                 np_negation_counts, np_textstringlengths)).transpose()
        return (np_text_list, features, np_text_labels)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def getData(attack):
    client = MongoClient("mongodb://tweeter:tweeter@psychic.cra.com/tweets?authMechanism=SCRAM-SHA-1")
    db = client.tweets
    cursor = db.historical_twitter.find()
    all_tweets = []
    index = 1
    with open('CRA_Historical_Anonymous_'+attack+'.json', 'w+') as f_gnip:
        for document in tqdm(cursor):
            document['_id']=''
            query_tag = document['gnip']['matching_rules'][0]['tag']
            if query_tag==attack:
                print document
                json.dump(document,f_gnip)
                f_gnip.write('\n')

    f_gnip.close()

def filterData(attack,startdate,enddate):
    f_filter = open('historical_twitter_darknetrelaunch_cratagged'+'_filter.json','w+')
    with open('historical_twitter_darknetrelaunch_cratagged'+'.json','r') as f_input:
        for line in tqdm(f_input):
            tt_json = json.loads(line)
            time_str = tt_json['postedTime']
            post_time = datetime.strptime(time_str,'%Y-%m-%dT%H:%M:%S.000Z')
            if post_time>=startdate and post_time<=enddate:
                json.dump(tt_json,f_filter)
                f_filter.write('\n')
    f_filter.close()

def getDataOPKKK():
    client = MongoClient("mongodb://tweeter:tweeter@psychic.cra.com/tweets?authMechanism=SCRAM-SHA-1")
    db = client.tweets
    cursor = db.historical_twitter_OPKKK.find()
    with open('CRA_Historical_Anonymous_OPKKK.json', 'w+') as f_gnip:
        for document in tqdm(cursor):
            document['_id']=''
            query_tag = document['gnip']['matching_rules'][0]['tag']
            if query_tag=='OPKKK':
                print document
                json.dump(document,f_gnip)
                f_gnip.write('\n')

    f_gnip.close()

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

def put_heirarchical():
    corpus = make_Corpus('txt_sentoken')
    word_polarity = get_polarity_words()
    labels = np.zeros(2000)
    labels[0:1000] = 0
    labels[1000:2000] = 1
    corpus2 = remove_high_sentiment_words(copy.copy(corpus), 2, 2, word_polarity)
    model1, vectorizer = train_SVM_model(corpus2, labels)
    all_inp=[]
    all=[]
    with open('all_sentiment_'+attack+'.json','r') as f_input:
        for line in tqdm(f_input):
            tt_json = json.loads(line)
            all_inp.append(tt_json['text'])
            all.append(tt_json)
    all_inp2 = remove_high_sentiment_words(copy.copy(all_inp), 0.35, 0.4, word_polarity)

    # Predict psychic data sentiment
    result2 = model1.predict(vectorizer.transform(all_inp2))
    f_result = open('all_sentiment_' + attack + '2.json', 'w+')
    for r in range(len(result2)):
        all[r]['sentiment2']=result2[r]
        json.dump(all[r], f_result)
        f_result.write('\n')


def sentimentAnalysis(attack):
    f_result = open('all_sentiment_'+attack+'.json','w+')
    model = pickle.load(open("./senti_model_1", 'rb'))
    all_tweets = []
    with open('historical_twitter_'+attack+'_filter.json','r') as f_input:
        for line in tqdm(f_input):
            tt_json = json.loads(line)
            entity = tt_json['actor']['displayName']
            tid = tt_json['id']
            timestamp = tt_json['postedTime']
            tweet = tt_json['body']
            tag  = tt_json["gnip"]["matching_rules"][0]["tag"]
            senti = model.obtain_document_list_probability([tweet])
            result_json = {'tid':tid,'tag':tag,'timestamp':timestamp,'sentiment1':senti[0][1],'user':entity,'text':tweet}
            json.dump(result_json,f_result)
            f_result.write('\n')
    print

def plotsentiment(attack):
    f_out = open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/historical_timelines_plot_'+attack+'.txt','w+')
    date_senti = dict()
    with open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/historical_timelines_'+attack+'.json','r') as f_senti:
        for line in tqdm(f_senti):
            senti_json = json.loads(line)
            timestamp = senti_json['timestamp']
            dateStr = timestamp[0:10]
            senti_score = senti_json['sentiment']
            if dateStr in date_senti:
                sentiments = date_senti[dateStr]
                sentiments.append(senti_score)
            else:
                date_senti[dateStr]=[senti_score]

    for k,v in date_senti.items():
        tmp_frequency = len(v)
        tmp_avg_senti = sum(v)/len(v)
        f_out.write(k+'\t'+str(tmp_frequency)+'\t'+str(tmp_avg_senti)+'\n')

    f_out.close()

def getalltags(attack,startdate,enddate):
    aa=set()
    f_out_FreedomHosting = open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/CRA_Historical_Anonymous_' + attack + '_FreedomHosting_filter.json', 'w+')
    f_out_OpDarknet = open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/CRA_Historical_Anonymous_' + attack + '_OpDarknet_filter.json', 'w+')
    with open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/historical_twitter_'+attack+'_cratagged.json','r') as f_input:
        for line in tqdm(f_input):
            tt_json = json.loads(line)
            time_str = tt_json['postedTime']
            post_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.000Z')
            if post_time >= startdate and post_time <= enddate:
                if 'FreedomHosting' in tt_json['cra_tags']:
                        json.dump(tt_json, f_out_FreedomHosting)
                        f_out_FreedomHosting.write('\n')
                if 'OpDarknet' in tt_json['cra_tags']:
                        json.dump(tt_json, f_out_OpDarknet)
                        f_out_OpDarknet.write('\n')
    f_out_FreedomHosting.close()
    f_out_OpDarknet.close()

def getalltagsAnon(attack,startdate,enddate):
    aa=set()
    f_out_FromAnonymous = open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/CRA_Historical_Anonymous_' + attack + '_FromAnonymous_filter.json', 'w+')
    with open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/historical_twitter_'+attack+'_anon_cratagged.json','r') as f_input:
        for line in tqdm(f_input):
            tt_json = json.loads(line)
            print tt_json
            time_str = tt_json['postedTime']
            post_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.000Z')
            if post_time >= startdate and post_time <= enddate:
                if 'FromAnonymous' in tt_json['cra_tags']:
                        print "HAHAHA"
                        json.dump(tt_json, f_out_FromAnonymous)
                        f_out_FromAnonymous.write('\n')
        f_out_FromAnonymous.close()

def plotlyplot(attack):
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go

    plotly.tools.set_credentials_file(username='anjoy92', api_key='4QKtyZysIeS1eQNHeXES')

    # Create random data with numpy
    import numpy as np
    dates = []
    freq = []
    senti = []
    output = []
    with open('/media/anjoy92/UBUNTU 16_0/Anonymous_Data/historical_timelines_plot_'+attack+'.txt','r') as f_senti:
        for line in f_senti:
            output.append(line.split('\t'))
    output.sort(key=lambda x: x[0])
    for a,b,c in output:
        dates.append(a)
        freq.append(b)
        senti.append(round(float(c),8))

    # Create traces
    trace0 = go.Scatter(
        x=dates,
        y=freq,
        mode='lines',
        name=attack+'_frequency'
    )
    # Create traces
    trace1 = go.Scatter(
        x=dates,
        y=senti,
        mode='lines',
        name=attack+'_sentiment'
    )

    data = [trace0]
    data2 = [trace1]

    py.iplot(data, filename=attack+'_frequency')
    py.iplot(data2, filename=attack+'_sentiment')

if __name__ == '__main__':
    # ***** DarknetRelaunch ****
    startdate = datetime.strptime('2017-01-13','%Y-%m-%d')
    enddate = datetime.strptime('2017-02-10','%Y-%m-%d')

    # ****** SingleGateway *****
    # startdate = datetime.strptime('2016-11-24','%Y-%m-%d')
    # enddate = datetime.strptime('2016-12-22','%Y-%m-%d')

    # ****** Comelec *****
    # startdate = datetime.strptime('2016-03-03','%Y-%m-%d')
    # enddate = datetime.strptime('2016-04-03','%Y-%m-%d')

    # ****** OPKKK *****
    # startdate = datetime.strptime('2015-10-07','%Y-%m-%d')
    # enddate = datetime.strptime('2015-11-04','%Y-%m-%d')



    attack = 'darknetrelaunch_cratagged'
    #filterData(attack,startdate,enddate)
    #sentimentAnalysis(attack)
    put_heirarchical()
    #plotsentiment(attack)
    # getData(attack)
    # getalltags(attack,startdate,enddate)
    # getalltagsAnon(attack,startdate,enddate)
    #plotlyplot(attack)