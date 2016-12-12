import sys
import math
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from cropping import crop
from shift_image import concatenate_data

try:
    cropped = sys.argv[2]
except IndexError:
    cropped = True

try:
    cross_validating = sys.argv[3]
except IndexError:
    cross_validating = False

# RANDOM FOREST CLASSIFIER
def random_forest_algo(X_train, y_train, X_test, y_test=None): 
    print 'Fitting a RandomForestClassifier'
    rf = RandomForestClassifier(n_estimators=1500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    rf.fit(X_train, y_train)
    output = rf.predict(X_test)
    if y_test:
        score = rf.score(X_test, y_test)
    else:
        score = 0
    return output, Counter(output), score 

def main():
    fraction_of_data_for_cross_val = 0.0
    if cross_validating:
        fraction_of_data_for_cross_val = 0.2
    
    try:
        filename = sys.argv[1]
    except IOError:
        filename = 'processed_train.csv'

    try:    
        training_data = pd.read_csv(filename)
    except IOError: 
        raw_train_data = pd.read_csv('train.csv')
        training_data = concatenate_data(raw_train_data)
    
    if cropped:
        rows = columns = int(math.sqrt(training_data.shape[1] -1))
        training_data = crop(training_data, rows, columns)
        
    data_features = training_data[training_data.columns.difference(['label'])]
    data_labels = training_data['label']
        
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_features, data_labels, fraction_of_data_for_cross_val, random_state=0)

    if cross_validating:
        output, spread, score = random_forest_algo(X_train, y_train, X_test, y_test)
        print 'Distribution of digits as recognized by classifier: ', spread
        print 'Cross-validation score = ', score
    else:
        test_data = pd.read_csv('test.csv')
        if cropped:
            test_data = crop(test_data, rows, columns)
        output, spread = random_forest_algo(X_train, y_train, test_data)
        print 'Distribution of digits as recognized by classifier: ', spread
        filename = 'digit_recognizer.csv'
        idx = range(1, test_data.shape[0] + 1)
        final_result = pd.DataFrame(data= output, index=idx, columns=['Label'])
        final_result.index.name = 'ImageId'
        final_result.to_csv(filename)

if __name__ == '__main__':
    main()
