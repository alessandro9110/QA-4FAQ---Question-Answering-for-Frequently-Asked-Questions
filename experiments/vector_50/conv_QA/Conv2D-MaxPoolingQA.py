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
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger
from collections import defaultdict
from scipy.stats.stats import pearsonr
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from keras import metrics


#LOAD TOKENIZER FILES
def loadSentences(filename):
    array = []
    with open(filename, encoding = "utf-8") as f:
        ar = []
        for line in f:
            if (str(line) != "[\n") & (str(line) != "]\n"):
                ar.append(re.sub('\n$', '', line))
                #print (line)
            if (str(line) == "]\n"):
                array.append(ar)
                ar = []
            
    f.close()
    return array


#SUBSTITUTE WORD WITH INDEX
def replace_matched_items(array, dictionary):
    for lst in array:
        for ind, item in enumerate(lst):
            lst[ind] = dictionary.get(item, item)       
                          

#CLEAN ZEROS
def cleaningZero(array):
    for elem in array:
        while 0 in elem: 
            elem.remove(0)
    

if __name__ == '__main__':

    
    K.os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    K.os.environ["CUDA_VISIBLE_DEVICES"]="0"


    #LOAD GENSIM MODEL WITH WORD2VEC 
    model_W2C = gensim.models.Word2Vec.load('Armillotta/model/vector_50/mymodelFull')

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('Armillotta/wiki-w2v/it-vectors.50.5.50.w2v', encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))


    # create a weight matrix for words in training docs
    vocab_size = len(model_W2C.wv.vocab)
    embedding_matrix = np.zeros((vocab_size, 50))
    for i in range(len(model_W2C.wv.vocab)):
        embedding_vector = embeddings_index.get(model_W2C.wv.index2word[i])
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    #ANSWER TRAINING
    original_A_train = loadSentences("Armillotta/tokenizer/tokenizerAnswer.txt")
    A_train = loadSentences("Armillotta/tokenizer/tokenizerAnswer.txt")

    #QUESTION  TRAINING
    original_Q_train = loadSentences("Armillotta/tokenizer/tokenizerQuestion.txt")
    Q_train = loadSentences("Armillotta/tokenizer/tokenizerQuestion.txt")


    #DATASET MANIPULATION FOR TRAINING
    random_idx = [random.randrange(0, len(original_Q_train)) for i in range(0, len(original_Q_train))]

    new = []
    for i in random_idx:#len(A_train)):
        new.append(original_Q_train[i])
    Q_train.extend(new)

    new = []
    for i in random_idx:#len(A_train)):
        new_random = random.randrange(0, len(random_idx))
        while (new_random == i):
            new_random = random.randrange(0, len(random_idx))
        new.append(original_A_train[new_random])
    A_train.extend(new)

    print(len(A_train))
    print(len(Q_train))


    #Y_train LABEL CLASS FOR EVALUATION TRAINING
    Y_train = []
    for i in range(0, (int(len(A_train)/2))):
        a=[]
        a.append(1)
        Y_train.append(a)
    for i in range(int(len(A_train)/2+1), len(A_train) + 1):
        a=[]
        a.append(0)
        Y_train.append(a)
    Y_train = np.array(Y_train)
    print(len(Q_train),len(A_train),len(Y_train))

                    

    #vocab - uso lo stesso vocab
    vocab = dict([(k, v.index) for k, v in model_W2C.wv.vocab.items()])

    #REPLACE A_TRAIN AND Q_TRAIN WITH VOCAB INDEX
    replace_matched_items(Q_train,vocab)
    cleaningZero(Q_train)
    replace_matched_items(A_train,vocab)
    cleaningZero(A_train)


    #PAD DOCUMENTS FOR QUESTION AND ANSWER
    max_len = max(len(max(A_train,key=len)), len(max(Q_train,key=len)))
    Q_train_padded = keras.preprocessing.sequence.pad_sequences(Q_train, padding='post', maxlen = max_len)
    A_train_padded = keras.preprocessing.sequence.pad_sequences(A_train, padding='post', maxlen = max_len)


    #FUNCTON FOR MATRIX MULTIPLICATION
    class OuterProduct(Layer):
        def __init__(self,name="OuterProduct",**kwargs):
            super().__init__()

        def build(self,input_shape):
            self.length = input_shape[0][1]
            super().build(input_shape)

        def call(self,x):
            a, b = x
    #        print(a.shape,b.shape,a[:, newaxis, :].shape,b[:, :, newaxis].shape)
            outerProduct = a[:, newaxis, :] * b[:, :, newaxis]
     #       print(outerProduct.shape)
            outerProduct = K.sum(outerProduct,axis=-1)
      #      print(outerProduct.shape)

            return outerProduct

        def compute_output_shape(self, input_shape):
            return (input_shape[0][0], input_shape[0][1], input_shape[0][1])
            
            

    #Save the model after every epoch. https://keras.io/callbacks/#modelcheckpoint
    checkpoint = ModelCheckpoint("Armillotta/keras_models/Q_A_Conv2D_MaxPool.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('Armillotta/history/CONV_Q-A/log.csv', append=True, separator=';')
    callbacks_list = [checkpoint,csv_logger]


    #LAYER EMBEDDING-BiLSTM FOR ANSWER
    A_input = Input(batch_shape=(None,A_train_padded.shape[1],), name='A_input')
    A = Embedding(len(vocab), output_dim=50, weights=[embedding_matrix],name="Embedding_A",trainable=True)(A_input)
    A_LSTM = Bidirectional(LSTM(50, dropout=0.5, recurrent_dropout=0.5,return_sequences=True),name="LSTM_A")(A)


    #LAYER EMBEDDING-BiLSTM FOR QUESTIONS
    Q_input = Input(batch_shape=(None,Q_train_padded.shape[1],),  name='Q_input')
    Q = Embedding(len(vocab),output_dim=50,name="Embedding_Q",trainable=True)(Q_input)
    Q_LSTM = Bidirectional(LSTM(50, dropout=0.5, recurrent_dropout=0.5,return_sequences=True),name="LSTM_QA")(Q)

    #PRODOTTO DELLE MATRICI EMB
    M1 = OuterProduct()([A, Q])
    M1 = Reshape((-1,A_train_padded.shape[1],Q_train_padded.shape[1]),name="M1_Embedding")(M1)
    M2 = OuterProduct()([A_LSTM, Q_LSTM])
    M2 = Reshape((-1,A_train_padded.shape[1],Q_train_padded.shape[1]),name="M2_LSTM")(M2)
    M12 = Concatenate(axis=1,name="M1xM2")([M1, M2])


    #CONVUTIONAL
    Convo = Conv2D(8, 3, activation='relu', data_format='channels_first',name="Convolutional")(M12)
    Convo = Dropout(.5)(Convo)

    #MAXPOOLING
    Pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_first',name="Max_Pooling")(Convo)
    Pool = Dropout(.5)(Pool)
    Pool = Flatten()(Pool)

    #MLP
    dense = Dense(200, activation='relu')(Reshape((-1,))(Pool))
    dense = Dropout(.5)(dense)
    dense= Dense(100, activation='relu')(dense)
    dense = Dropout(.5)(dense)

    output = Dense(1,activation='sigmoid')(dense)

    model = Model(inputs=[Q_input,A_input], outputs=output)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',metrics.mean_squared_error])

    #model.summary(line_length=200)
    #print("Model Ready...")
    
    #PRINT MODEL LAYERS ON FILE.PNG
    plot_model(model, to_file='Armillotta/image/Q_A_Conv2D_MaxPool.png', show_shapes=True, show_layer_names=True)


    history = model.fit([Q_train_padded,A_train_padded],Y_train, batch_size=25, epochs=200, validation_split=0.1,callbacks = callbacks_list)



