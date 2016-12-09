import pandas as pd
import logging

print 'Reading in training data \n'
training_data = pd.read_csv('train.csv')

# Helper function to shift image by one column to the left or to the right
def imageshifterbyonecolumn(data, to_pixel, from_pixel):
    print 'Shifting image by one column from {} to {} followed by subsequent columns'.format(from_pixel, to_pixel)
    shifted_data = pd.DataFrame(columns=data.columns.values) #new dataframe that will contain data of the shifted image
    shifted_data['label'] = data['label'] 
    shifted_data[shifted_data.columns.difference(['label', to_pixel])] = \
        data[data.columns.difference(['label', from_pixel])] #shifting the pixel values by one column
    shifted_data.loc[:, to_pixel] = data.loc[:, from_pixel] #wrapping around the 'from_pixel' values into the 'to_pixel' column
    return shifted_data


# Function to create one concatenated dataframe containing the raw data and new data from the shifted images
def shift_image():
    left_shifted_training_data = imageshifterbyonecolumn(training_data, 'pixel783', 'pixel0') #shifting images by one column to the left
    right_shifted_training_data = imageshifterbyonecolumn(training_data, 'pixel0', 'pixel783') #shifting images by one column to the right
    return pd.concat([training_data, left_shifted_training_data, right_shifted_training_data]) 


filename = 'processed_train.csv'
print 'Writing processed training data into {}'.format(filename)
final_training_data = shift_image()
final_training_data.to_csv(filename, index=False)

