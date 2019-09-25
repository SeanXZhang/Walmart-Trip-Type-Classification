import pickle  # for saving/loading binary files (serializing/deserializing)
import time

import numpy as np
import pandas as pd
import calendar
from sklearn.metrics import log_loss, accuracy_score, classification_report


def prepare_data(file_path, sample_size=None, target_col='TripType'):

    start = time.time()
    print('Reading training data...')
    data_train = pd.read_csv(file_path).replace("MENS WEAR","MENSWEAR")
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))

    # the original dataset is quite large. You can randomly sample a good amount of rows from it for this task.
    if sample_size is not None:
        data_train = data_train.sample(sample_size)
    
    start = time.time()
    print('Processing training data...')

    dept_list = sorted(list(data_train.DepartmentDescription.dropna().unique()))
    
    weekdays = list(calendar.day_name)
    dept_list_sum = dict.fromkeys(dept_list, np.sum)
    weekday_dict = dict.fromkeys(weekdays, np.max)
    feature_dict = {"TripType": np.max, 'NumItems': np.sum, 'Return': np.max}
    feature_dict = {**feature_dict, **weekday_dict, **dept_list_sum}
    
    data_new = transform_data(data_train, feature_dict, weekdays, dept_list)
    
    data_new = add_category_counts(data_new, dept_list)
    
    fln_dummies = fineline_dummies(data_train)
    data_new = data_new.join(fln_dummies)
    
    upc_dummies = Upc_dummies(data_train)
    data_new = data_new.join(upc_dummies)
    
    X = data_new.drop(target_col, axis=1)

    trip_types = sorted(data_train.TripType.unique())
    trip_types_map = dict(zip(trip_types, np.arange(0, len(trip_types))))
    y = data_new.TripType.map(trip_types_map)
    
    end = time.time()
    print(f'Finished processing training data in %.3f seconds' % (end - start))
    
    return X, y


def transform_data(data, feature_dict, weekdays, dept_list):
    dummies = pd.get_dummies(data.Weekday)
    data[dummies.columns] = dummies
    
    dummies = pd.get_dummies(data.DepartmentDescription)
    dummies = dummies.apply(lambda x: x*data["ScanCount"])
    data[dummies.columns] = dummies 

    data.loc[data.ScanCount < 0, 'Return'] = 1
    data.loc[data.Return != 1, 'Return'] = 0
    
    data = data.rename(columns={"ScanCount":"NumItems"})
    
    grouped = data.groupby("VisitNumber")
    grouped = grouped.aggregate(feature_dict)
    data = grouped[["TripType", "NumItems", "Return"] + weekdays + dept_list]

    return data


def add_category_counts(data, dept_list):
    alist = []
    for array in np.asarray(data.loc[:, dept_list[0]:]):
        count = 0
        count = sum(x > 0 for x in array)
        alist.append(count)
    cat_counts = pd.DataFrame(alist)
    cat_counts = cat_counts.rename(columns={0:"CategoryCount"})
    cat_counts = cat_counts.set_index(data.index)
    data.insert(3, 'CategoryCounts', cat_counts)
    return data


def fineline_dummies(data):
    values = data.FinelineNumber
    counts = values.value_counts()
    mask = values.isin(counts[counts > 500].index)
    values[~mask] = "-"
    dummies = pd.get_dummies(values).drop('-', axis=1)

    dummies.columns = ['fln_'+str(col) for col in dummies.columns]
    dummies = pd.concat([dummies, data.VisitNumber], axis=1)
    dummies = dummies.groupby("VisitNumber")
    dummies = dummies.aggregate(np.sum)
    return dummies


def Upc_dummies(data):
    values = data.Upc
    counts = values.value_counts()
    mask = values.isin(counts[counts > 300].index)
    values[~mask] = "-"
    dummies = pd.get_dummies(values).drop('-', axis=1)

    dummies.columns = ['upc_'+str(col) for col in dummies.columns]
    dummies = pd.concat([dummies, data.VisitNumber], axis=1)
    dummies = dummies.groupby("VisitNumber")
    dummies = dummies.aggregate(np.sum)
    return dummies


def evaluate_prediction(model_name, sclf, X_test, y_test):
    #print(classification_report(y_test, predictions))
    y_pred = sclf.predict(X_test)
    y_proba = sclf.predict_proba(X_test)
    print('[{}] Accuracy: {}, Log Loss: {}'.format(model_name, accuracy_score(y_test, y_pred), log_loss(y_test, y_proba)))


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_binary(path):
    return pickle.load(open(path, 'rb'))


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        # print(results)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
