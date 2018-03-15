import os
import sys
import certifi
import unicodedata
import pandas as pd
import numpy as np
import re
import csv
import math
import requests
import time
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import gensim
import keras
import h5py
import pydot
import os
import sklearn
import operator
import time

from keras.layers.merge import Add
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.layers import Conv2D, MaxPooling2D, Embedding, Merge, LSTM, Bidirectional, GlobalMaxPooling1D, Dot, Lambda, Dropout
from keras.layers import Average, dot,Permute, Lambda,Layer, add, Concatenate, Dense, LSTM, Input, concatenate, merge, Add, Reshape, Flatten
from keras.models import Model, load_model
from keras.models import Sequential
from numpy import newaxis
from keras.models import model_from_json
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from collections import defaultdict
from scipy.stats.stats import pearsonr
from keras import backend as K
from sklearn.model_selection import StratifiedKFold

K.os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
K.os.environ["CUDA_VISIBLE_DEVICES"]="0"


# FUNCTON FOR MATRIX MULTIPLICATION
class OuterProduct(Layer):
    def __init__(self, name="OuterProduct", **kwargs):
        super().__init__()

    def build(self, input_shape):
        self.length = input_shape[0][1]
        super().build(input_shape)

    def call(self, x):
        a, b = x
        #        print(a.shape,b.shape,a[:, newaxis, :].shape,b[:, :, newaxis].shape)
        outerProduct = a[:, newaxis, :] * b[:, :, newaxis]
        #       print(outerProduct.shape)
        outerProduct = K.sum(outerProduct, axis=-1)
        #      print(outerProduct.shape)

        return outerProduct

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][1])


# SUBSTITUTE WORD WITH INDEX
def replace_matched_items(array, dictionary):
    for lst in array:
        for ind, item in enumerate(lst):
            lst[ind] = dictionary.get(item, item)


# CLEAN ZEROS
def cleaningZero(array):
    for elem in array:
        while 0 in elem:
            elem.remove(0)


target = 25


# SAVE ANSWER INDEXS
def getAnswerIDX(pred):
    x = {}
    y = []
    for i in range(0, len(pred)):
        x[i] = float(pred[i])
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    for j in range(0, len(sorted_x)):
        y.append(sorted_x[j][0])
    return y[0:target]


def getAnswer(resp, test):
    lst = resp
    ans = []
    for i in lst:
        ans.append(test[i])
    return ans


# SPLIT ANSWERS
def printAnswer(a):
    answers = []

    for j in range(0, len(a)):
        sentence = ""
        for i in a[j]:
            sentence = sentence + " " + i
        answers.append(sentence)
    return answers


# LOAD TOKENIZER FILES
def loadSentences(filename):
    array = []
    with open(filename, encoding='utf8') as f:
        ar = []
        for line in f:
            if (str(line) != "[\n") & (str(line) != "]\n"):
                ar.append(re.sub('\n$', '', line))
                # print (line)
            if (str(line) == "]\n"):
                array.append(ar)
                ar = []

    f.close()
    return array

if __name__ == '__main__':

    # ANSWERING TEST
    original_A_test = loadSentences("/home/inversi/FAQProject/Armillotta/tokenizer/tokenizerAnswer.txt")
    A_test = loadSentences("/home/inversi/FAQProject/Armillotta/tokenizer/tokenizerAnswer.txt")

    # QUESTION TEST--> 3 QUESTIONS
    original_Q_test = loadSentences("/home/inversi/FAQProject/Armillotta/tokenizer/tokenizerQTest.txt")
    Q_test = loadSentences("/home/inversi/FAQProject/Armillotta/tokenizer/tokenizerQTest.txt")

    # CONV2D - MAXPOOLING MODEL
    product = load_model("Q_A_Conv2D_MaxPool.h5",
                         custom_objects={'OuterProduct': OuterProduct})

    # LOAD GENSIM MODEL WITH WORD2VEC
    model_W2C = gensim.models.Word2Vec.load('mymodelFull')

    # vocab - uso lo stesso vocab
    vocab = dict([(k, v.index) for k, v in model_W2C.wv.vocab.items()])

    Respond = []
    Ids = []

    for i in range(0, len(original_Q_test)):
        start_time = time.time()
        Q_test = original_Q_test[i]
        Q_test = [Q_test] * len(A_test)

        # REPLACE A_TEST AND Q_TEST WITH VOCAB INDEX
        replace_matched_items(A_test, vocab)
        cleaningZero(A_test)
        replace_matched_items(Q_test, vocab)
        cleaningZero(Q_test)

        # pad documents with original max len
        max_len = max(len(max(A_test, key=len)), len(max(Q_test, key=len)))
        Q_test_padded = keras.preprocessing.sequence.pad_sequences(Q_test, padding='post', maxlen=max_len)
        A_test_padded = keras.preprocessing.sequence.pad_sequences(A_test, padding='post', maxlen=max_len)

        for j in range(0, 1):

            # CONV2D MAXPOOLING FOR QUESTION 1
            prodQ1 = product.predict([Q_test_padded, A_test_padded])
            ansID = getAnswerIDX(prodQ1)
            AnswersConvQ1 = getAnswer(ansID, original_A_test)

            # SAVE CONV2D MAXPOOLING ANSWERS IN A TXT FILE
            Answers = printAnswer(AnswersConvQ1)

            print(i + 1)
            print("--- %s seconds ---" % (time.time() - start_time))

            with open('outcome_final.csv', 'a', newline='', encoding="utf-8") as csvfile:
                fieldnames = ['idx', 'risposte']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

                for i in range(len(ansID)):
                    writer.writerow({'idx': ansID[i], 'risposte': Answers[i]})
