import sys
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from cropping import crop
from shift_image import concatenate_data

# DEFINE CONSTANTS
FRACTION_OF_TEST_DATA = 0.0
ROWS = 28
COLUMNS = 28

try:
    cropped = sys.argv[2]
except IndexError:
    cropped = True

# READ IN DATA FROM CSV
print 'Reading in training and test data \n'


test_data = pd.read_csv('test.csv')

try:
    filename = sys.argv[1]
except:
    filename = 'processed_train.csv'
    
try:    
    training_data = pd.read_csv(filename)
except IOError: 
    raw_train_data = pd.read_csv('train.csv')
    training_data = concatenate_data(raw_train_data)

if cropped:
    training_data = crop(training_data, ROWS, COLUMNS)
    test_data = crop(test_data, ROWS, COLUMNS)
    
data_features = training_data[test_data.columns.values]
data_labels = training_data['label']

# SPLIT DATA INTO TRAIN-TEST BASED ON FRACTION DEFINED IN CONSTANTS, IDENTIFY DATA FEATURES AND LABELS
X_train, X_test, y_train, y_test = model_selection.train_test_split(data_features, data_labels, FRACTION_OF_TEST_DATA, random_state=0)

# RANDOM FOREST CLASSIFIER
def random_forest_algo(X_train, y_train, X_test):#, y_test): #when cross-validating the model, uncomment the y_test parameter
    print 'Fitting a RandomForestClassifier'
    rf = RandomForestClassifier(n_estimators=1500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    rf.fit(X_train, y_train)
    output = rf.predict(X_test)
    return output, Counter(output) #, rf.score(X_test, y_test) #when cross-validating, uncomment the rf.score to check performance of model

# USING THE RANDOM FOREST CLASSIFIER ON THE TEST DATA
output, spread = random_forest_algo(X_train, y_train, test_data)
print spread

# UNCOMMENT BELOW TO CROSS-VALIDATE THE MODEL. THIS WILL USE THE TRAINING DATA SPLIT INTO TRAINING AND CROSS-VALIDATING
# DATASETS TO CHECK THE PERFORMANCE OF THE MODEL USE THE INBUILT SCORE METHOD.
# output, spread, score = random_forest_algo(X_train, y_train, X_test, y_test)
# print spread
# print score

# CREATING THE CSV FILE WITH THE OUTPUT DATA IN THE FORMAT SPECIFIED FOR SUBMISSION TO KAGGLE
filename = 'digit_recognizer.csv'
print 'Writing output data for submission to Kaggle to {}'.format(filename)
idx = range(1, test_data.shape[0] + 1)
final_result = pd.DataFrame(data= output, index=idx, columns=['Label'])
final_result.index.name = 'ImageId'
final_result.to_csv(filename)

