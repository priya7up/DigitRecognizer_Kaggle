import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from image_cropped import crop_or_not

# DEFINE CONSTANTS
FRACTION_OF_TEST_DATA = 0.0
ROWS = 28
COLUMNS = 28
cropped = True

# READ IN DATA FROM CSV
print 'Reading in training and test data \n'
training_data = pd.read_csv('processed_train.csv')
test_data = pd.read_csv('test.csv')


# SPLIT DATA INTO TRAIN-TEST BASED ON FRACTION DEFINED IN CONSTANTS, IDENTIFY DATA FEATURES AND LABELS
def test_train_data(training_data, test_data, fraction_of_test_data, rows, columns, cropped):
    print 'Creating train and test datasets'
    data_features, data_labels, final_test_data = crop_or_not(training_data, test_data, rows, columns, cropped)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_features, data_labels,
                                                                         test_size=fraction_of_test_data,
                                                                         random_state=0)
    return X_train, X_test, y_train, y_test, final_test_data


# CALL TEST-TRAIN FUNCTION TO GET TEST AND TRAINING DATA SETS
X_train, X_test, y_train, y_test, final_test_data = test_train_data(training_data, test_data, FRACTION_OF_TEST_DATA, ROWS, COLUMNS, cropped)

# RANDOM FOREST CLASSIFIER
def random_forest_algo(X_train, y_train, X_test):#, y_test): #when cross-validating the model, uncomment the y_test parameter
    print 'Fitting a RandomForestClassifier'
    rf = RandomForestClassifier(n_estimators=1500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    rf.fit(X_train, y_train)
    output = rf.predict(X_test)
    return output, Counter(output) #, rf.score(X_test, y_test) #when cross-validating, uncomment the rf.score to check performance of model

# USING THE RANDOM FOREST CLASSIFIER ON THE TEST DATA
output, spread = random_forest_algo(X_train, y_train, final_test_data)
print spread

# UNCOMMENT BELOW TO CROSS-VALIDATE THE MODEL. THIS WILL USE THE TRAINING DATA SPLIT INTO TRAINING AND CROSS-VALIDATING
# DATASETS TO CHECK THE PERFORMANCE OF THE MODEL USE THE INBUILT SCORE METHOD.
# output, spread, score = random_forest_algo(X_train, y_train, X_test, y_test)
# print spread
# print score

# CREATING THE CSV FILE WITH THE OUTPUT DATA IN THE FORMAT SPECIFIED FOR SUBMISSION TO KAGGLE
filename = 'digit_recognizer.csv'
print 'Writing output data for submission to Kaggle to {}'.format(filename)
idx = range(1,28001)
final_result = pd.DataFrame(data= output, index=idx, columns=['Label'])
final_result.index.name = 'ImageId'
final_result.to_csv(filename)

