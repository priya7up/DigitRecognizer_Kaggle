import math
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from cropping import crop
from shift_image import concatenate_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('cropped', nargs='?', help='Crop the images?', default=True)
parser.add_argument('filename', nargs='?', help='Save shift_image data in this file',default='processed_train.csv')
parser.add_argument('cross_validating', nargs='?', help='Checking cross_validation score?', default=False)
args = parser.parse_args()


def random_forest_algo(X_train, y_train, X_test, y_test=None): 
    """ Implements a random forest classifier and returns the predicted labels, distribution of classified labels and a score
    evaluating the performance of the classifier (only if cross_validating) """
    
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                max_leaf_nodes=None, bootstrap=True, n_jobs=1, random_state=None, verbose=0, 
                                warm_start=False, class_weight=None)
    rf.fit(X_train, y_train)
    output = rf.predict(X_test)
    if y_test:
        score = rf.score(X_test, y_test)
    else:
        score = 0
    return output, Counter(output), score 

def main():
    fraction_of_data_for_cross_val = 0.0
    if args.cross_validating:
        fraction_of_data_for_cross_val = 0.2
    
    try:    
        training_data = pd.read_csv(args.filename)
    except IOError: 
        raw_train_data = pd.read_csv('train.csv')
        training_data = concatenate_data(raw_train_data)
    
    if args.cropped:
        rows = columns = int(math.sqrt(training_data.shape[1] -1))
        training_data = crop(training_data, rows, columns)
        
    data_features = training_data[training_data.columns.difference(['label'])]
    data_labels = training_data['label']
        
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_features, data_labels, fraction_of_data_for_cross_val, random_state=0)

    if args.cross_validating:
        output, spread, score = random_forest_algo(X_train, y_train, X_test, y_test)
        print 'Distribution of digits as recognized by classifier: ', spread
        print 'Cross-validation score = ', score
    else:
        test_data = pd.read_csv('test.csv')
        if cropped:
            test_data = crop(test_data, rows, columns)
        output, spread = random_forest_algo(X_train, y_train, test_data)
        print 'Distribution of digits as recognized by classifier: ', spread
        idx = range(1, test_data.shape[0] + 1)
        final_result = pd.DataFrame(data= output, index=idx, columns=['Label'])
        final_result.index.name = 'ImageId'
        final_result.to_csv('digit_recognizer.csv')

if __name__ == '__main__':
    main()
