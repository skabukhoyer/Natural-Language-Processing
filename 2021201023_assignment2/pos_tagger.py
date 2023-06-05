# import necessary libraries
import string
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import conll2000
from tensorflow import keras
# import seaborn as sns

# from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from conllu import parse

train_data = open('./UD_English-Atis/en_atis-ud-train.conllu')
dev_data = open('./UD_English-Atis/en_atis-ud-dev.conllu')
test_data = open('./UD_English-Atis/en_atis-ud-test.conllu')

train_data = train_data.read()
dev_data = dev_data.read()
test_data = test_data.read()

train_sentences = parse(train_data)
dev_sentences = parse(dev_data)
test_sentences = parse(test_data)


train_X = []
train_Y = []

dev_X = []
dev_Y = []

test_X = []
test_Y = []

for sentence in train_sentences:
    X_sentence = []
    Y_sentence = []
    for entity in sentence:         
        X_sentence.append(entity['form'])  # entity[0] contains the word
        Y_sentence.append(entity['upos'])  # entity[1] contains corresponding tag
        
    train_X.append(X_sentence)
    train_Y.append(Y_sentence)

for sentence in dev_sentences:
    X_sentence = []
    Y_sentence = []
    for entity in sentence:         
        X_sentence.append(entity['form'])  # entity[0] contains the word
        Y_sentence.append(entity['upos'])  # entity[1] contains corresponding tag
        
    dev_X.append(X_sentence)
    dev_Y.append(Y_sentence)

for sentence in test_sentences:
    X_sentence = []
    Y_sentence = []
    for entity in sentence:         
        X_sentence.append(entity['form'])  # entity[0] contains the word
        Y_sentence.append(entity['upos'])  # entity[1] contains corresponding tag
        
    test_X.append(X_sentence)
    test_Y.append(Y_sentence)

word_tokenizer = Tokenizer(oov_token = '<OOV>')                      # instantiate tokeniser
word_tokenizer.fit_on_texts(train_X + test_X + dev_X)                    # fit tokeniser on data
tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(train_Y + test_Y + dev_Y)


MAX_SEQ_LENGTH = 100

tag2id = tag_tokenizer.word_index
tag2id['unk'] = 0
id2tag = {}
for tag,idx in tag2id.items():
    id2tag[idx] = tag


def convert_int_categorical(test_Y_pred):
    test_Y_pred = test_Y_pred.tolist()

    for i in range(len(test_Y_pred)):
        sent = test_Y_pred[i]

        for j in range(len(sent)):
            word = sent[j]

            maxv = 0
            maxk = 0

            for k in range(len(word)):    
                if (word[k] > maxv):
                    maxv = word[k]
                    maxk = k

            test_Y_pred[i][j] = maxk

    test_Y_pred = np.asarray(test_Y_pred)
    return test_Y_pred


def predict_sentence_POS_Tags(sentence,model,word_tokenzier,pos_tag_dict_rev):
    cleaned_sentence = sentence.translate(str.maketrans('','',string.punctuation))
    testing_sentence = cleaned_sentence.lower().split()
    test_list = []
    test_list.append(testing_sentence)
    testing_sentence_encoded = word_tokenizer.texts_to_sequences(test_list)
    testing_sentence_padded = pad_sequences(testing_sentence_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
    testing_sentence_pred = model.predict(testing_sentence_padded)
    
    testing_sentence_pred = convert_int_categorical(testing_sentence_pred)
    sent_len = len(test_list[0])
    for word,idx in zip(test_list[0],testing_sentence_pred[0][100-sent_len:]):
        print("{}\t{}".format(word, id2tag[idx].upper()))


sentence = input('')

model = keras.models.load_model('./model')
predict_sentence_POS_Tags(sentence,model,word_tokenizer,id2tag)


