from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import string
from tqdm import tqdm
import json
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    tokens = [w.translate(None, string.punctuation) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)

#resu=model.predict_classes(Xtest)


def get_psychic_data():
    psyc_data = {}
    with open('newsdata.json') as data_file:
        psyc_data = json.load(data_file)

    all_news = []
    for news in psyc_data:
        if news:
            all_news.append(news['article_text'])
    return all_news

all=[]
all_inp=[]
# attack='darknetrelaunch_cratagged'
# with open('all_sentiment_'+attack+'2.json','r') as f_input:
#     for line in tqdm(f_input):
#         tt_json = json.loads(line)
#         all_inp.append(tt_json['text'])
#         all.append(tt_json)
all_inp=get_psychic_data()

print train_docs
train_docs=all_inp
print train_docs

encoded_docs = tokenizer.texts_to_sequences(train_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
resu=model.predict_classes(Xtest)
for r in resu:
    print r,

cnt_po=0
cnt_ne=0

for r in resu:
    if r[0]==1:
        cnt_po+=1
    elif r[0]==0:
        cnt_ne+=1

print cnt_po,cnt_ne

f_result = open('all_sentiment_newsdata.json', 'w+')
for r in range(len(resu)):
    print resu[r][0]
    print all[r]
    all[r]['sentiment3']=int(resu[r][0])
    json.dump(all[r], f_result)
    f_result.write('\n')
print resu
# evaluate
#loss, acc = model.evaluate(Xtest, ytest, verbose=0)
#print('Test Accuracy: %f' % (acc * 100))