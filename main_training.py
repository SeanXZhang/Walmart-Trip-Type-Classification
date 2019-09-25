import warnings
import time
import os
from config import *
from models.model import *
from utils.utils import *
from keras.models import load_model
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    X, y = prepare_data(os.path.join(DATA_SUBDIR, 'train.csv'))

    start = time.time()
    print('Train test splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    end = time.time()
    print(f'Finished train test splitting data in %.3f seconds' % (end - start))

    save_binary((X_train, X_test, y_train, y_test), os.path.join(RESULTS_DATA_SUB, 'data.h5'))
    
    model_train = 'XGB_CLF'
    train = False
    
    if train:
        
        print(f'Training %s...' % model_train)
        
        if model_train == 'Neural Network':
            saved_NN_model_path = NeuralNet(X_train, y_train, layer=1)
            saved_NN_model = load_model(saved_NN_model_path)

            loss, acc = saved_NN_model.evaluate(X_test, to_categorical(y_test))
    
            print('[{}] {}： {}, {}: {}'.format(model_train, saved_NN_model.metrics_names[0], loss, 
                  saved_NN_model.metrics_names[1], acc))
        
        if model_train == 'NN Stacking':
            sclf, model = NN_Stacking(X_train, X_test, y_train, y_test)
            evaluate_prediction(model_train, sclf, X_test, y_test)
            path = os.path.join(RESULTS_MODEL_SUB, model_train+'.pkl')
            save_binary(model, path)
        
        if model_train == 'NN Stacking with LR':
            sclf, model = NN_Stacking_with_LR(X_train, X_test, y_train, y_test)
            evaluate_prediction(model_train, sclf, X_test, y_test)
            path = os.path.join(RESULTS_MODEL_SUB, model_train+'.pkl')
            save_binary(model, path)
        
        if model_train == 'NN Stacking with XGB':
            sclf, model = NN_Stacking_with_XGB(X_train, X_test, y_train, y_test)
            evaluate_prediction(model_train, sclf, X_test, y_test)
            path = os.path.join(RESULTS_MODEL_SUB, model_train+'.pkl')
            save_binary(model, path)
        
        if model_train == 'RFC':
            rfc, model = RFC(X_train, X_test, y_train, y_test)
            evaluate_prediction(model_train, rfc, X_test, y_test)
            path = os.path.join(RESULTS_MODEL_SUB, model_train+'.pkl')
            save_binary(model, path)
        
        if model_train == 'XGB_CLF':
            xgbc, model = XGB(X_train, X_test, y_train, y_test)
            evaluate_prediction(model_train, xgbc, X_test, y_test)
            path = os.path.join(RESULTS_MODEL_SUB, model_train+'.pkl')
            save_binary(model, path)
        
    
    model_tune = 'RFC_RS'
    tune = True
    
    if tune:
        
        print(f'Parameter tuning by random grid')
        
        scoring = ['neg_log_loss', 'accuracy']
        
        batch_size = [500, 1000, 2000, 3000]
        
        if model_tune == 'NN_1HiddenLayer_RS':
            nodes1 = np.arange(350, 600, 50)
            dropout1 = np.arange(0.1, 0.5, 0.1)
            
            params = dict(nodes_l1=nodes1, dropout_l1=dropout1, batch_size=batch_size)
            
            best_estimator = NeuralNetRandomizedSearchCV(X, y, params, scoring, layer=1)
            path = os.path.join(RESULTS_MODEL_SUB, model_tune+'.pkl')
            save_binary(best_estimator, path)
        
        if model_tune == 'NN_2HiddenLayers_RS':
            nodes1 = np.arange(350, 550, 50)
            nodes2 = np.arange(100, 350, 50)

            dropout1 = np.arange(0.1, 0.5, 0.1)
            dropout2 = np.arange(0.1, 0.5, 0.1)
                        
            params = dict(nodes_l1=nodes1, nodes_l2=nodes2, dropout_l1=dropout1, dropout_l2=dropout2, batch_size=batch_size)
            
            best_estimator = NeuralNetRandomizedSearchCV(X, y, params, scoring, layer=2)
            path = os.path.join(RESULTS_MODEL_SUB, model_tune+'.pkl')
            save_binary(best_estimator, path)
        
        if model_tune == 'NN_3HiddenLayers_RS':
            nodes1 = np.arange(350, 550, 50)
            nodes2 = np.arange(150, 350, 50)
            nodes3 = np.arange(60, 150, 30)

            dropout1 = np.arange(0.1, 0.5, 0.1)
            dropout2 = np.arange(0.1, 0.5, 0.1)
            dropout3 = np.arange(0.1, 0.5, 0.1)
                        
            params = dict(nodes_l1=nodes1, nodes_l2=nodes2, nodes_l3=nodes3, 
                          dropout_l1=dropout1, dropout_l2=dropout2, dropout_l3=dropout3, batch_size=batch_size)
            
            best_estimator = NeuralNetRandomizedSearchCV(X, y, params, scoring, layer=3)
            path = os.path.join(RESULTS_MODEL_SUB, model_tune+'.pkl')
            save_binary(best_estimator, path)
            
        if model_tune == 'RFC_RS':
            params = {'bootstrap': [True, False], 'max_depth': np.arange(50, 90, 10), 'min_samples_split': [3, 5, 7], 
                      'min_samples_leaf': [2, 4], 'max_features': ['sqrt'], 'n_estimators': [50]}
            
            best_estimator = RFC_RS(X, y, params, scoring)
            path = os.path.join(RESULTS_MODEL_SUB, model_tune+'.pkl')
            save_binary(best_estimator, path)
            
        if model_tune == 'XGB_RS':
            params = {'objective': ['multi:softmax'], 'max_depth': [5, 7, 9, 11], 'n_estimators': [50], 
                      'min_child_weight': [1, 5, 10], 'gamma': [0, 2, 4, 6, 8], 'subsample': [0.6, 0.8, 1.0], 
                      'colsample_bytree': [0.6, 0.8, 1.0]}

            best_estimator = XGB_RS(X, y, params, scoring)
            path = os.path.join(RESULTS_MODEL_SUB, model_tune+'.pkl')
            save_binary(best_estimator, path)
            