import pandas as pd
import logging

def main():
    logging.basicConfig(filename='digit_recognizer.log', level=logging.INFO)
    logging.info('Reading in training data \n')
    training_data = pd.read_csv('train.csv')

    # Helper function to shift image by one column to the left or to the right
    def imageshifterbyonecolumn(data, to_pixel, from_pixel):
        logging.info('Shifting image by one column from {} to {} followed by subsequent columns'.format(from_pixel, to_pixel))
        """ Creating a new dataframe that will contain data of the shifted image, 
            the number of pixels in the shifted image is the same as the original image """
        shifted_data = pd.DataFrame(columns=data.columns.values)
        shifted_data['label'] = data['label'] 
        shifted_data[shifted_data.columns.difference(['label', to_pixel])] = \
            data[data.columns.difference(['label', from_pixel])] 
        shifted_data.loc[:, to_pixel] = data.loc[:, from_pixel] 
        return shifted_data


    # Function to create one concatenated dataframe containing the raw data and new data from the shifted images
    def shift_image():
        """ Shifting images by one column to the left """
        left_shifted_training_data = imageshifterbyonecolumn(training_data, 'pixel783', 'pixel0') 
        """ Shifting images by one column to the right """
        right_shifted_training_data = imageshifterbyonecolumn(training_data, 'pixel0', 'pixel783') 
        return pd.concat([training_data, left_shifted_training_data, right_shifted_training_data]) 


    filename = 'processed_train.csv'
    logging.info('Writing processed training data into {}'.format(filename))
    final_training_data = shift_image()
    final_training_data.to_csv(filename, index=False)

if __name__ == '__main__':
    main()

