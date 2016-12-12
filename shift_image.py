import pandas as pd
import logging
import sys

def create_shifted_image(data, to_pixel, from_pixel):
        """ Helper function to shift image by one column to the left or to the right """
        logging.info('Shifting image by one column from %s to %s followed by subsequent columns',from_pixel, to_pixel)
        #Creating a new dataframe that will contain data of the shifted image, the number of pixels in the shifted image 
        #is the same as the original image
        shifted_data = pd.DataFrame(columns=data.columns.values) 
        shifted_data['label'] = data['label'] 
        shifted_data[shifted_data.columns.difference(['label', to_pixel])] = \
            data[data.columns.difference(['label', from_pixel])] 
        shifted_data.loc[:, to_pixel] = data.loc[:, from_pixel] 
        return shifted_data


def concatenate_data(data):
    """ Function to create one concatenated dataframe containing the raw data and new data from the shifted images """
    last_pixel = 'pixel' + str(data.shape[1] - 2)
    #Shifting images by one column to the left    
    left_shifted_training_data = create_shifted_image(data, last_pixel, 'pixel0') 
    #Shifting images by one column to the right    
    right_shifted_training_data = create_shifted_image(data, 'pixel0', last_pixel) 
    return pd.concat([data, left_shifted_training_data, right_shifted_training_data]) 


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Reading in training data \n')
    training_data = pd.read_csv('train.csv')
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = 'processed_train.csv'
    logging.info('Writing processed training data into %s', filename)
    final_training_data = concatenate_data(training_data)
    final_training_data.to_csv(filename, index=False)

if __name__ == '__main__':
    main()

