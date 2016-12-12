def pixels_to_crop(rows, columns):
    """ Function takes in the total number of rows and columns in the image and calculates the pixels to remove in order 
        to crop the image by two rows/columns from each side """
    remove_pixels = set()
    for i in range(rows):
        for j in range(columns):
            if i == 0 or i == 1 or i == (columns-2) or i == (columns-1):
                remove_pixels.add(i + columns*j)
            if j == 0 or j == 1 or j == (rows-2) or j == (rows-1):
                remove_pixels.add(i + rows*j)
    output = sorted(list(remove_pixels))
    return ['pixel' + str(x) for x in output]


# def crop(training_data, test_data, rows, columns, cropped):
#     if cropped:
#         cropped_training_data = training_data[training_data.columns.difference(cropping(rows, columns))]
#         final_test_data = test_data[test_data.columns.difference(cropping(rows, columns))]
#         print 'Using cropped data'
#         data_labels = cropped_training_data['label']
#         data_features = cropped_training_data[final_test_data.columns.values]
#     else:
#         print 'Using non-cropped data'
#         final_test_data = test_data
#         data_labels = training_data['label']
#         data_features = training_data[final_test_data.columns.values]
#     return data_features, data_labels, final_test_data
def crop(data, rows, columns):
    return data[data.columns.difference(pixels_to_crop(rows, columns))]
