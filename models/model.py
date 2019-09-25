from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, classification_report
from mlxtend.classifier import StackingClassifier
from datetime import datetime
import os
from config import *

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 


def create_model_NN_1HiddenLayer(input_dim, nodes_l1=512, dropout_l1=0.3):
    model = Sequential()
    model.add(Dense(output_dim=nodes_l1, activation="relu", input_dim=input_dim))
    model.add(Dropout(dropout_l1))
            
    model.add(Dense(output_dim=38, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    
    return model


def create_model_NN_2HiddenLayers(input_dim, nodes_l1=400, nodes_l2=120, dropout_l1=0.3, dropout_l2=0.2):
    model = Sequential()
    model.add(Dense(output_dim=nodes_l1, activation="relu", input_dim=input_dim))
    model.add(Dropout(dropout_l1))
    model.add(Dense(output_dim=nodes_l2, activation="relu"))
    model.add(Dropout(dropout_l2))
            
    model.add(Dense(output_dim=38, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    
    return model


def create_model_NN_3HiddenLayers(input_dim, nodes_l1=500, nodes_l2=200, nodes_l3=80, 
                                  dropout_l1=0.3, dropout_l2=0.2, dropout_l3=0.2):
    model = Sequential()
    model.add(Dense(output_dim=nodes_l1, activation="relu", input_dim=input_dim))
    model.add(Dropout(dropout_l1))
    model.add(Dense(output_dim=nodes_l2, activation="relu"))
    model.add(Dropout(dropout_l2))
    model.add(Dense(output_dim=nodes_l2, activation="relu"))
    model.add(Dropout(dropout_l3))
            
    model.add(Dense(output_dim=38, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    
    return model


def create_model_NN_4HiddenLayers(input_dim, nodes_l1=512, nodes_l2=256, nodes_l3=128, nodes_l4=64, 
                                  dropout_l1=0.3, dropout_l2=0.3, dropout_l3=0.2, dropout_l4=0.2):
    model = Sequential()
    model.add(Dense(output_dim=nodes_l1, activation="relu", input_dim=input_dim))
    model.add(Dropout(dropout_l1))
    model.add(Dense(output_dim=nodes_l2, activation="relu"))
    model.add(Dropout(dropout_l2))
    model.add(Dense(output_dim=nodes_l3, activation="relu"))
    model.add(Dropout(dropout_l3))
    model.add(Dense(output_dim=nodes_l4, activation="relu"))
    model.add(Dropout(dropout_l3))
            
    model.add(Dense(output_dim=38, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    
    return model


def NeuralNet(X, y, layer=1):   
    model_name = ''
    input_dim=len(X.columns)
    
    if layer == 1:
        model_name = 'NN_1HiddenLayer'
        clf = KerasClassifier(build_fn=create_model_NN_1HiddenLayer, input_dim=input_dim, epochs=30, batch_size=3000, verbose=2)

    if layer == 2:
        model_name = 'NN_2HiddenLayers'
        clf = KerasClassifier(build_fn=create_model_NN_2HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
        
    if layer == 3:
        model_name = 'NN_3HiddenLayers'
        clf = KerasClassifier(build_fn=create_model_NN_3HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
        
    if layer == 4:
        model_name = 'NN_4HiddenLayers'
        clf = KerasClassifier(build_fn=create_model_NN_4HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)

    filepath = os.path.join(RESULTS_MODEL_SUB, model_name+'_best_model.h5') 
    callbacks = [EarlyStopping(monitor="val_acc", patience=20), 
                 ModelCheckpoint(filepath=filepath, monitor="val_acc", save_best_only=True)]
    
    start = datetime.now()
    print('Fitting Neural Net: %s...' % model_name)
    clf.fit(X, y, validation_split=0.25, epochs=100, callbacks=callbacks)
    end = datetime.now()
    print(f'Finished fitting Neural Net in %s seconds' % str(end-start))
    
    print('Saved model to path: %s' % filepath)

    return filepath


def NN_Stacking(X_train, X_test, y_train, y_test):
    model_name = 'NN Stacking'
    input_dim = len(X_train.columns)
    
    #create_model_NN_1HiddenLayer(input_dim=38*4, nodes_l1=200, dropout_l1=0.2)
    clf1 = KerasClassifier(build_fn=create_model_NN_1HiddenLayer, input_dim=38*4, epochs=30, batch_size=1000, verbose=2)
    clf2 = KerasClassifier(build_fn=create_model_NN_2HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    clf3 = KerasClassifier(build_fn=create_model_NN_3HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    clf4 = KerasClassifier(build_fn=create_model_NN_4HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    
    sclf = StackingClassifier(classifiers=[clf2, clf2, clf3, clf4], use_probas=True, average_probas=False, meta_classifier=clf1)
    
    start = datetime.now()
    print('Fitting %s...' % model_name)
    model = sclf.fit(X_train, y_train)
    end = datetime.now()
    print('Finished fitting {} in {} seconds'.format(model_name, str(end-start)))

    return sclf, model


def NN_Stacking_with_LR(X_train, X_test, y_train, y_test):
    model_name = 'NN Stacking with LR'
    input_dim = len(X_train.columns)
    
    clf1 = KerasClassifier(build_fn=create_model_NN_1HiddenLayer, input_dim=input_dim, epochs=30, batch_size=1000, verbose=2)
    clf2 = KerasClassifier(build_fn=create_model_NN_2HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    clf3 = KerasClassifier(build_fn=create_model_NN_3HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    clf4 = KerasClassifier(build_fn=create_model_NN_4HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    
    lr = LogisticRegression(C=0.4, solver='lbfgs', multi_class='multinomial', random_state=42)
    
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], use_probas=True, average_probas=False, meta_classifier=lr)
        
    start = datetime.now()
    print('Fitting %s...' % model_name)
    model = sclf.fit(X_train, y_train)
    end = datetime.now()
    print('Finished fitting {} in {} seconds'.format(model_name, str(end-start)))

    return sclf, model


def NN_Stacking_with_XGB(X_train, X_test, y_train, y_test):
    model_name = 'NN Stacking with XGB'
    input_dim = len(X_train.columns)
    
    clf1 = KerasClassifier(build_fn=create_model_NN_1HiddenLayer, input_dim=input_dim, epochs=30, batch_size=1000, verbose=2)
    clf2 = KerasClassifier(build_fn=create_model_NN_2HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    clf3 = KerasClassifier(build_fn=create_model_NN_3HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    clf4 = KerasClassifier(build_fn=create_model_NN_4HiddenLayers, input_dim=input_dim, epochs=15, batch_size=1000, verbose=2)
    
    xgbc = XGBClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', 
                         max_depth=50, bootstrap=False, random_state=42)
    
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], use_probas=True, average_probas=False, meta_classifier=xgbc)
        
    start = datetime.now()
    print('Fitting %s...' % model_name)
    model = sclf.fit(X_train, y_train)
    end = datetime.now()
    print('Finished fitting {} in {} seconds'.format(model_name, str(end-start)))

    return sclf, model


def RFC(X_train, X_test, y_train, y_test):
    model_name = 'Random Forest Classifier'
    rfc = RandomForestClassifier(bootstrap=False, max_depth=80, min_samples_split=6, min_samples_leaf=2, 
                                 max_features='sqrt', n_estimators=200, random_state=42)
    
    start = datetime.now()
    print('Fitting %s...' % model_name)
    model = rfc.fit(X_train, y_train)
    end = datetime.now()
    print('Finished fitting {} in {} seconds'.format(model_name, str(end-start)))

    return rfc, model


def XGB(X_train, X_test, y_train, y_test):
    model_name = 'XGB Classifier'

    xgbc = XGBClassifier(objective='multi:softmax', n_estimators=100, subsample=0.6, min_child_weight=5, max_depth=11, 
                         gamma=4, colsample_bytree=0.6, random_state=42, verbosity=3)
    
    start = datetime.now()
    print('Fitting %s...' % model_name)
    model = xgbc.fit(X_train, y_train)
    end = datetime.now()
    print('Finished fitting {} in {} seconds'.format(model_name, str(end-start)))

    return xgbc, model


def NeuralNetRandomizedSearchCV(X, y, params, scoring, layer=1):
    input_dim = len(X.columns)
    cv=StratifiedKFold(n_splits=4)
    model_name = ''
    
    if layer == 1:
        model_name = 'NN 1HiddenLayer'
        clf = KerasClassifier(build_fn=create_model_NN_1HiddenLayer, input_dim=input_dim, epochs=15, verbose=2)
    if layer == 2:
        model_name = 'NN 2HiddenLayers'
        clf = KerasClassifier(build_fn=create_model_NN_2HiddenLayers, input_dim=input_dim, epochs=15, verbose=2)
    if layer == 3:
        model_name = 'NN 3HiddenLayers'
        clf = KerasClassifier(build_fn=create_model_NN_3HiddenLayers, input_dim=input_dim, epochs=20, verbose=2)
        
    rs = RandomizedSearchCV(estimator=clf, param_distributions=params, cv=cv, n_iter=3, scoring=scoring, 
                            refit='accuracy', random_state=42, n_jobs=1, verbose=2)
    
    start = datetime.now()
    print('Starting RandomizedSearchCV for %s...' % model_name)
    rs.fit(X, y)
    end = datetime.now()
    print('Finished RandomizedSearchCV for {} in {} seconds'.format(model_name, str(end-start)))
    print('Best Acc: {}, Best Params: {}'.format(rs.best_score_, rs.best_params_))
    
    return rs.best_estimator_


def RFC_RS(X, y, params, scoring):
    model_name = 'RandomForestClassifier'
    
    rfc = RandomForestClassifier(n_estimators=50, random_state=42)
    
    gs = GridSearchCV(rfc, param_grid=params, scoring=scoring, cv=4, refit='accuracy', verbose=1, n_jobs=-1)
    start = datetime.now()
    print('Starting GridSearchCV for %s...' % model_name)
    gs.fit(X, y)
    end = datetime.now()
    print('Finished GridSearchCV for {} in {} seconds'.format(model_name, str(end-start)))
    print('Best Acc: {}, Best Params: {}'.format(gs.best_score_, gs.best_params_))
    
    return gs.best_estimator_


def XGB_RS(X, y, params, scoring):
    model_name = 'RandomForestClassifier'
    
    xgbc = XGBClassifier(n_estimators=50, random_state=42)
    
    rs = RandomizedSearchCV(xgbc, param_distributions=params, scoring=scoring, cv=4, n_iter=50, refit='accuracy', 
                            return_train_score=True, random_state=42, n_jobs=-1, verbose=3)
    start = datetime.now()
    print('Starting RandomizedSearchCV for %s...' % model_name)
    rs.fit(X, y)
    end = datetime.now()
    print('Finished RandomizedSearchCV for {} in {} seconds'.format(model_name, str(end-start)))
    print('Best Acc: {}, Best Params: {}'.format(rs.best_score_, rs.best_params_))
    
    return rs.best_estimator_
