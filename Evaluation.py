import numpy as np
from gensim.models import Word2Vec
import KaneMetrics
from gensim.test.utils import common_texts, get_tmpfile
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import TimeDistributed
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
       
class ResultsSave:
    classes = None
    y_pred = None
    y_true = None

class Evaluator:
    current_x_train = None
    current_x_test = None
    current_y_train = None
    current_y_test = None
    current_y_pred = None
    current_y_true = None
    current_classes = None
    label_binarizer = None
    token = None
    metrics = {}
    labelAccuracy = {}
    outputAcc = []
    learningRateHistory = []
    last_lr = None
    km = None

    def saveResults( self, exp ):
        f = open( exp + ".log","w+")
        f.write( classification_report( self.label_binarizer.inverse_transform(self.current_y_true), self.label_binarizer.inverse_transform(self.current_y_pred) ))

        f.write( "\n\n" );
        for label in self.labelAccuracy:
            f.write( label + " " + str(np.average( self.labelAccuracy[label] )) + " " + str(np.std( self.labelAccuracy[label])) + "\n" )

        res = ResultsSave()
        res.y_true = self.label_binarizer.inverse_transform( self.current_y_true )
        res.y_pred = self.label_binarizer.inverse_transform( self.current_y_pred )
        res.classes = self.current_classes

        pickle.dump( res, open( exp + ".p", "wb" ) );
        f.close()

    def plotLossAccuracy( self, history, experiment ):
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(history.history['loss'])
        plt.plot(history.history['accuracy'])
        plt.title('Model Loss & Accuracy - ' + experiment)
        plt.ylabel('')
        plt.xlabel('Epoch')
        plt.legend(['Loss', 'Accuracy'], loc='upper right')
        plt.savefig( 'figures/' + experiment + '_loss_acc.png' )
        return plt

    def plotLearningRate( self, experiment = '' ):
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(self.learningRateHistory)
        plt.title('Learning Rate - ' + experiment)
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.show()
    
    def plotLoss( self, history, experiment = '' ):
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss - ' + experiment)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig( 'figures/' + experiment + '_loss.png' )
        return plt

    def plotAccuracy( self, history, experiment = '' ):
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy - ' + experiment)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.show()

    def plotConfusionMatrix( self, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
    
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        #print(cm)    
    
        fig, ax = plt.subplots(figsize=(8,8),dpi=600)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()
        return ax

    def evaluateModel( self, x_test, y_test, model, data, k = 0 ): 
        y_pred = model.predict( x_test, verbose=0 )
        y_pred[y_pred >= .5] = 1
        y_pred[y_pred < .5] = 0
        y_pred = y_pred.astype( int )
        scores = model.evaluate( x_test, y_test, verbose=0 )
   
        if self.current_y_pred is None: 
            self.current_y_pred = y_pred
            self.current_y_true = y_test
        else:
            self.current_y_pred = np.concatenate( (self.current_y_pred, y_pred ), axis=0)
            self.current_y_true = np.concatenate( (self.current_y_true, y_test ), axis=0)

        self.metrics['recall'] = ( float( recall_score( y_test, y_pred, average='weighted' ) ) )
        self.metrics['label_accuracy'] = {}
  
        print( " " ) 
        print( "Metrics:" ) 
        print( "Recall: %.2f%%" % ( float( recall_score( y_test, y_pred, average='weighted' ) ) * 100 ) )
        print( "SciKit Accuracy: %.2f%%" % ( float( accuracy_score( y_test, y_pred ) ) * 100 ) )
        labelCnt = {}
        labelMatch = {}
        badPredictionCnt = {}

        y_test_orig = self.label_binarizer.inverse_transform( y_test )
        y_pred_orig = self.label_binarizer.inverse_transform( y_pred )
        print( f1_score( y_test_orig, y_pred_orig, average=None) ) 
        print( np.unique( y_test_orig ) )
        print( classification_report( y_test_orig, y_pred_orig ) )
        print( "Accuracy by Label: ") 
        for i in range( len( y_test_orig ) ):
            yLabel = y_pred_orig[i]

            if yLabel not in labelCnt:
                labelCnt[yLabel] = 0
            if yLabel not in labelMatch:
                labelMatch[yLabel] = 0
            if yLabel not in badPredictionCnt:
                badPredictionCnt[yLabel] = 0
            
            if y_pred_orig[i] == y_test_orig[i]:
                labelMatch[yLabel] = labelMatch[yLabel] + 1
            else:
                badPredictionCnt[yLabel] = badPredictionCnt[yLabel] + 1

            labelCnt[yLabel] = labelCnt[yLabel] + 1
        for label in labelCnt:
            print( label, ": %.2f%%" % (labelMatch[label]/labelCnt[label]*100) )
            self.metrics['label_accuracy'][label] = labelMatch[label]/labelCnt[label]
            if label not in self.labelAccuracy:
                self.labelAccuracy[label] = []
            self.labelAccuracy[label].append( labelMatch[label]/labelCnt[label] )

        print( badPredictionCnt )

        if self.km is None:
            self.km = KaneMetrics.KaneMetrics()

        self.km.addMetric( self.metrics, "WCVD", k  )
 
        return model, scores
