import Models
import Features
import Evaluation

import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def ETFIDFD():
    import Endava
    data = Endava.Endava()
    __TFIDFD( data, "ETFIDFD" )

def WTFIDFD():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    __TFIDFD( data, "WTFIDFD" )

def ECVD():
    import Endava
    data = Endava.Endava()
    __CVD( data, "ECVD" )

def WCVD():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    __CVD( data, "WCVD" )

def WGLD():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    filepath = 'glove.42B.300d.txt'
    __EmbeddedD( data, 'WGLD', filepath )

def EGLD():
    import Endava
    data = Endava.Endava()
    filepath = 'glove.42B.300d.txt'
    __EmbeddedD( data, 'EGLD', filepath )

def WW2VD():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    filepath = 'enwiki_20180420_300d.txt'
    __EmbeddedD( data, 'WW2VD', filepath )

def WW2VCNN():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    filepath = 'enwiki_20180420_300d.txt'
    __EmbeddedCNN( data, 'WW2VCNN', filepath )

def EW2VCNN():
    import Endava
    data = Endava.Endava()
    filepath = 'enwiki_20180420_300d.txt'
    __EmbeddedCNN( data, 'EW2VCNN', filepath )

def WCVCNN():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    filepath = None
    __EmbeddedCNN( data, 'WCVCNN', filepath )

def ECVCNN():
    import Endava
    data = Endava.Endava()
    filepath = None
    __EmbeddedCNN( data, 'ECVCNN', filepath )

def EGLCNN():
    import Endava
    data = Endava.Endava()
    filepath = 'glove.42B.300d.txt'
    __EmbeddedCNN( data, 'EGLCNN', filepath )

def EW2VD():
    import Endava
    data = Endava.Endava()
    filepath = 'enwiki_20180420_300d.txt'
    __EmbeddedD( data, 'EW2VD', filepath )

def __EmbeddedRNN( data, exp, filepath, network ):
    kn = Evaluation.Evaluator()    
    X,Y,y_raw = Features.getSamples( kn, data )
    data.maxWords = 10000
    kf = StratifiedKFold( n_splits=10, shuffle=True )
    k = 0 

    for train, test in kf.split( X, y_raw ):
        print( "K-Fold: " + str( k + 1 ) );
        x_train_raw, x_test_raw = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        Models.embedding, x_train, x_test = Features.getEmbedded( x_train_raw, x_test_raw, 
            y_train, y_test, y_raw, filepath, kn )
        batches = [64,218]
        neurons = [100,200]
        dropouts = [2,3]
        param_grid = dict( batch_size=batches, neuron=neurons, dropout=dropouts, output_size=[len(y_train[0])] )
        model = None
        if network == 'lstm':
            model = KerasClassifier(build_fn=Models.create_lstm_model, epochs=30, verbose=2)
        else:
            model = KerasClassifier(build_fn=Models.create_gru_model, epochs=30, verbose=2)

        y_ints = [y.argmax() for y in y_train]
        cweights = class_weight.compute_class_weight( 'balanced', np.unique( y_ints ), y_ints )

        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(x_train, y_train, class_weight=cweights)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        model, scores = kn.evaluateModel( x_test, y_test, grid.best_estimator_.model, data, k )
        k = k + 1 

    kn.saveResults( exp )

def __EmbeddedCNN( data, exp, filepath ):
    kn = Evaluation.Evaluator()    
    X,Y,y_raw = Features.getSamples( kn, data )
    data.maxWords = 10000
    kf = StratifiedKFold( n_splits=10, shuffle=True )
    k = 0 

    for train, test in kf.split( X, y_raw ):
        print( "K-Fold: " + str( k + 1 ) );
        x_train_raw, x_test_raw = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        Models.embedding, x_train, x_test = Features.getEmbedded( x_train_raw, x_test_raw, 
            y_train, y_test, y_raw, filepath, kn )
        batches = [16]
        pool_sizes = [3,5]
        layer_sizes = [128, 64]
        param_grid = dict( batch_size=batches, pool_size=pool_sizes, layer_size=layer_sizes, output_size=[len(y_train[0])] )
        model = KerasClassifier(build_fn=Models.create_cnn_model, epochs=15, verbose=2)

        y_ints = [y.argmax() for y in y_train]
        cweights = class_weight.compute_class_weight( 'balanced', np.unique( y_ints ), y_ints )

        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(x_train, y_train, class_weight=cweights)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        model, scores = kn.evaluateModel( x_test, y_test, grid.best_estimator_.model, data, k )
        k = k + 1 

    kn.saveResults( exp )

def WStack():
    import Wesleyan
    data = Wesleyan.Wesleyan()
    kn = Evaluation.Evaluator()    
    exp = "WStack"
    filepath = 'enwiki_20180420_300d.txt'

    X,Y,y_raw = Features.getSamples( kn, data )
    data.maxWords = 10000
    kf = StratifiedKFold( n_splits=10, shuffle=True )
    k = 0 
    for train, test in kf.split( X, y_raw ):
        print( "K-Fold: " + str( k + 1 ) );
        x_train_raw, x_test_raw = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        y_ints = [y.argmax() for y in y_train]
        cweights = class_weight.compute_class_weight( 'balanced', np.unique( y_ints ), y_ints )
        
        Models.embedding, x_train, x_test = Features.getEmbedded( x_train_raw, x_test_raw, 
            y_train, y_test, y_raw, filepath, kn )

        y_pred_train = None
        y_pred_test = None       
        def do_cnn(): 
            model1 = Models.create_cnn_model( pool_size=3, layer_size=128, output_size=len(y_train[0]) )
            model1.fit(x_train, y_train, epochs=15, verbose=2, batch_size=32, class_weight=cweights)

            y_pred_train = model1.predict( x_train, verbose=0 )
            y_pred_test = model1.predict( x_test, verbose=0 )
            model1 = None

        do_cnn()
 
        model2 = Models.create_ann_model( dropout=3, denseSize=512, output_length=len(y_train[0]) )
        model2.fit(x_train, y_train, batch_size=64, verbose=2, epochs=30, class_weight=cweights)
        
        y_pred_train2 = model2.predict( x_train, verbose=0 )
        y_pred_test2 = model2.predict( x_test, verbose=0 )
        model2 = None
        
        x_train, x_test = Features.getCVvectors( x_train_raw, x_test_raw, data )
        model3 = Models.create_ann_model( batch_size=64, denseSize=1024,
            dropout=3, input_length=[len(x_train[0])],
            output_length=[len(y_train[0])] )
        model3.fit(x_train, y_train, epochs=100, class_weight=cweights)

        y_pred_train3 = model3.predict( x_train, verbose=0 )
        y_pred_test3 = model3.predict( x_test, verbose=0 )

        new_x_train = np.stack( (y_pred_train, y_pred_train2, y_train_3 ), axis=-1)
        new_x_test = np.stack( (y_pred_test, y_pred_test2, y_test_3 ), axis=-1)
        
        model = Models.create_stack_model( input_size=len(new_x_train[0]), output_size=len(y_train[0]) )
        history = model.fit(new_x_train, y_train, epochs=100, verbose=2, batch_size=128, class_weight=cweights )

        model, scores = kn.evaluateModel( new_x_test, y_test, model, data, k )
        k = k + 1 
    
    kn.saveResults( exp )

def __EmbeddedD( data, exp, filepath ):
    kn = Evaluation.Evaluator()    
    X,Y,y_raw = Features.getSamples( kn, data )
    data.maxWords = 10000
    kf = StratifiedKFold( n_splits=10, shuffle=True )
    k = 0 

    for train, test in kf.split( X, y_raw ):
        print( "K-Fold: " + str( k + 1 ) );
        x_train_raw, x_test_raw = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        Models.embedding, x_train, x_test = Features.getEmbedded( x_train_raw, x_test_raw, 
            y_train, y_test, y_raw, filepath, kn )
        denseSizes = [512, 256]
        batches = [32, 16]
        dropouts = [2,3]
        param_grid = dict( batch_size=batches, denseSize=denseSizes, 
            dropout=dropouts,
            output_length=[len(y_train[0])] )
        model = KerasClassifier(build_fn=Models.create_ann_model, epochs=100, verbose=2)

        y_ints = [y.argmax() for y in y_train]
        cweights = class_weight.compute_class_weight( 'balanced', np.unique( y_ints ), y_ints )

        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(x_train, y_train, class_weight=cweights)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        model, scores = kn.evaluateModel( x_test, y_test, grid.best_estimator_.model, data, k )
        k = k + 1 

    kn.saveResults( exp )

def __TFIDFD( data, exp ):
    kn = Evaluation.Evaluator()    
    X,Y,y_raw = Features.getSamples( kn, data )
    data.maxWords = 10000
    kf = StratifiedKFold( n_splits=10, shuffle=True )
    k = 0 

    for train, test in kf.split( X, y_raw ):
        print( "K-Fold: " + str( k + 1 ) );
        x_train_raw, x_test_raw = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        x_train, x_test = Features.getTFIDFvectors( x_train_raw, x_test_raw, data )
        denseSizes = [512,1024]
        batches = [64,128]
        dropouts = [2,3]
        param_grid = dict( batch_size=batches, denseSize=denseSizes, 
            dropout=dropouts, input_length=[len(x_train[0])],
            output_length=[len(y_train[0])] )
        model = KerasClassifier(build_fn=Models.create_ann_model, epochs=30, verbose=2)

        y_ints = [y.argmax() for y in y_train]
        cweights = class_weight.compute_class_weight( 'balanced', np.unique( y_ints ), y_ints )

        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(x_train, y_train, class_weight=cweights)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        model, scores = kn.evaluateModel( x_test, y_test, grid.best_estimator_.model, data, k )
        k = k + 1 

    kn.saveResults( exp )

def __CVD( data, exp ):
    kn = Evaluation.Evaluator()    
    X,Y,y_raw = Features.getSamples( kn, data )
    data.maxWords = 10000
    kf = StratifiedKFold( n_splits=10, shuffle=True )
    k = 0 

    for train, test in kf.split( X, y_raw ):
        print( "K-Fold: " + str( k + 1 ) );
        x_train_raw, x_test_raw = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        x_train, x_test = Features.getCVvectors( x_train_raw, x_test_raw, data )
        denseSizes = [512,1024]
        batches = [64,128]
        dropouts = [2,3]
        param_grid = dict( batch_size=batches, denseSize=denseSizes, 
            dropout=dropouts, input_length=[len(x_train[0])],
            output_length=[len(y_train[0])] )
        model = KerasClassifier(build_fn=Models.create_ann_model, epochs=30, verbose=2)

        y_ints = [y.argmax() for y in y_train]
        cweights = class_weight.compute_class_weight( 'balanced', np.unique( y_ints ), y_ints )

        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(x_train, y_train, class_weight=cweights)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        model, scores = kn.evaluateModel( x_test, y_test, grid.best_estimator_.model, data, k )
        k = k + 1 

    kn.saveResults( exp )

# Complete
#EW2VCNN()

EGLCNN()





# TODO
# ECVD()
# ETFIDFD()
# EW2VD()
# EGLD()
# EBERT()
# EGLL()
# EGLG()
# EW2VL()
# EW2VG()
# ECVG()
# ECVL()
# ECVCNN()

# WCVD()
# WTFIDFD()
# WW2VD()
# WGLD()
# WBERT()
# WGLL()
# WGLG()
# WW2VL()
# WW2VG()
# WCVG()
# WCVL()
# WCVCNN()
# WW2VCNN()
# WGLCNN()

