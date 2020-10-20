import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from gensim.test.utils import common_texts, get_tmpfile
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

#X,Y,y_raw = kn.getSamples( data )
#data.maxWords = 10000

def getSamples( evaluator, data ):
    Y_data, corpus = data.getRawSamples()
    evaluator.current_classes = np.unique( Y_data )
    evaluator.label_binarizer = preprocessing.LabelBinarizer()
    Y = evaluator.label_binarizer.fit_transform( Y_data ) 
    return np.array( corpus ), np.array( Y ), Y_data

def vectorizeSamples( x_train_raw, x_test_raw, vectorizer ):
    vectorizer.fit( x_train_raw )
    vector_features = vectorizer.transform( x_train_raw )
    x_train = vector_features.toarray()
    x_test = vectorizer.transform( x_test_raw ).toarray()
    return x_train, x_test

def getTFIDFvectors( x_train_raw, x_test_raw ):
    vectorizer = TfidfVectorizer( max_features=data.maxWords, min_df=0.0, sublinear_tf=True )    
    return vectorizeSamples( x_train_raw, x_test_raw, vectorizer )

def getCVvectors( x_train_raw, x_test_raw, data ):
    vectorizer = CountVectorizer( min_df=50, max_features=data.maxWords, stop_words='english' )    
    return vectorizeSamples( x_train_raw, x_test_raw, vectorizer )

def getEmbedded( x_train_raw, x_test_raw, y_train, y_test, y_raw, filepath, kn ):
    kn.current_classes = np.unique( y_raw )

    if filepath is None:
        token = keras.preprocessing.text.Tokenizer()
        token.fit_on_texts( x_train_raw )
        kn.token = token

        x_train_raw = token.texts_to_sequences( x_train_raw )
        x_test_raw = token.texts_to_sequences( x_test_raw )

        x_test = keras.preprocessing.sequence.pad_sequences(x_test_raw, maxlen=300, padding='post')
        x_train = keras.preprocessing.sequence.pad_sequences(x_train_raw, maxlen=300, padding='post')
        vocab_size = len( token.word_index ) + 1 
        return Embedding(vocab_size,sequenceLength, input_length=300,trainable=False), x_train, x_test

    sequenceLength = 300 
    embedding_dim = 300 
    token = keras.preprocessing.text.Tokenizer()
    token.fit_on_texts( x_train_raw )
    kn.token = token
    vocab_size = len( token.word_index ) + 1 

    x_train_raw = token.texts_to_sequences( x_train_raw )
    x_test_raw = token.texts_to_sequences( x_test_raw )

    x_test = keras.preprocessing.sequence.pad_sequences(x_test_raw, maxlen=sequenceLength, padding='post')
    x_train = keras.preprocessing.sequence.pad_sequences(x_train_raw, maxlen=sequenceLength, padding='post')
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    word_index = token.word_index

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return Embedding(vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequenceLength, trainable=False), x_train, x_test
