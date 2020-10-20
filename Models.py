import numpy as np
from gensim.models import Word2Vec
import KaneMetrics
from gensim.test.utils import common_texts, get_tmpfile
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Flatten
from keras.layers import TimeDistributed
from sklearn.model_selection import GridSearchCV
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, GRU, LSTM, SpatialDropout1D, Dropout, GRU, Bidirectional, MaxPooling1D
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels 
from sklearn.utils import class_weight
from keras import optimizers
from keras import callbacks
import keras
import random
import pickle
from keras.callbacks import LearningRateScheduler

embedding = None

def create_stack_model( output_size = None, input_size = None ):
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    model.add(Dense(output_size, activation='softmax'))
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] )
    print(model.summary())
    return model

def create_gru_model( neuron=None, dropout=None, output_size=None ):
    keras.backend.clear_session()
    dropout = dropout / 10
    model = Sequential()
    model.add(embedding)
    model.add(Dropout(dropout))
    model.add(GRU(neuron, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(neuron))
    model.add(Dropout(dropout))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def create_lstm_model( neuron=None, dropout=None, output_size=None ):
    keras.backend.clear_session()
    dropout = dropout / 10
    model = Sequential()
    model.add(embedding)
    model.add(Dropout(dropout))
    model.add(LSTM(neuron, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(neuron))
    model.add(Dropout(dropout))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def create_cnn_model( pool_size = 5, layer_size = 256, output_size = None ):
    keras.backend.clear_session()
    model = Sequential()
    model.add(embedding)
    model.add(Conv1D(layer_size, pool_size, activation='relu', padding="same"))
    model.add(MaxPooling1D(pool_size, padding="same"))  
    model.add(Conv1D(layer_size, pool_size, activation='relu', padding="same"))
    model.add(MaxPooling1D(pool_size, padding="same"))  
    model.add(Conv1D(layer_size, pool_size, activation='relu', padding="same"))
    model.add(MaxPooling1D(5 * pool_size, padding="same"))  
    model.add( Flatten() )
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def create_ann_model( dropout = 3, denseSize = 512, input_length = None, output_length = None ):
    keras.backend.clear_session()
    model = Sequential()
    dropout = dropout / 10
    if embedding is None:
        model.add(Dropout(dropout, input_shape=(input_length,)))
    else:
        model.add(embedding)
        model.add(Dropout(dropout))

    model.add(Dense(denseSize, activation='relu'))
    model.add(Dropout(dropout))

    if embedding is not None:
        model.add( Flatten() )

    model.add(Dense(int(denseSize / 2), activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(output_length, activation='softmax'))

    opt = optimizers.RMSprop(lr=0.001, rho=0.9, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] )
    print(model.summary())
    return model
