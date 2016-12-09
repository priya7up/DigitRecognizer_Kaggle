import pandas as pd

# READ IN DATA FROM CSV
print 'Reading in training data \n'
training_data = pd.read_csv('train.csv')

# HELPER FUNCTION TO SHIFT IMAGE BY ONE COLUMN TO THE LEFT OR TO THE RIGHT
def imageshifterbyonecolumn(data, to_pixel, from_pixel):
    print 'Shifting image by one column from {} to {} followed by subsequent columns'.format(from_pixel, to_pixel)
    shifted_data = pd.DataFrame(columns=data.columns.values) #new dataframe that will contain data of the shifted image
    shifted_data['label'] = data['label'] #copying over the label from the original images
    shifted_data[shifted_data.columns.difference(['label', to_pixel])] = \
        data[data.columns.difference(['label', from_pixel])] #moving the pixel values by one column
    shifted_data.loc[:, to_pixel] = data.loc[:, from_pixel] #wrapping around the 'from_pixel' values into the 'to_pixel' column
    return shifted_data


# FUNCTION TO CREATE ONE CONCATENATED DATAFRAME CONTAINING THE RAW DATA AND NEW DATA OF THE SHIFTED IMAGES
def shift_image():
    left_shifted_training_data = imageshifterbyonecolumn(training_data, 'pixel783', 'pixel0') #shifting images by one column to the left
    right_shifted_training_data = imageshifterbyonecolumn(training_data, 'pixel0', 'pixel783') #shifting images by one column to the right
    return pd.concat([training_data, left_shifted_training_data, right_shifted_training_data]) #concatenating the three dataframes


# WRITING THE NEW DATAFRAME INTO A CSV FILE.
filename = 'processed_train.csv'
print 'Writing processed training data into {}'.format(filename)
final_training_data = shift_image()
final_training_data.to_csv(filename, index=False)

