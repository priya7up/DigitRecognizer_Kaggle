# # FUNCTION TO CROP IMAGE IF NEEDED. ROWS AND COLUMNS PARAMETERS ARE THE TOTAL NUMBER OF ROWS AND COLUMNS OF PIXELS
# # IN THE IMAGE. FUNCTION i + 28*j IS DEFINED IN THE DIGIT RECOGNIZER PROBLEM
def cropping(rows, columns):
    print 'Cropping image'
    remove_pixels = []
    for i in range(rows):
        for j in range(columns):
            if i == 0 or i == 1 or i == (columns-2) or i == (columns-1):
                remove_pixels.append(i + columns*j)
            if j == 0 or j == 1 or j == (rows-2) or j == (rows-1):
                remove_pixels.append(i + rows*j)
    output = sorted(set(remove_pixels))
    final_output = []
    for item in output:
        str_item = 'pixel' + str(item)
        final_output.append(str_item)
    return final_output


# DEFINE FEATURE VECTORS AND LABELS BASED ON WHETHER WE WANT TO WORK WITH CROPPED OR FULL IMAGE
def crop_or_not(training_data, test_data, rows, columns, cropped):
    if cropped:
        cropped_training_data = training_data[training_data.columns.difference(cropping(rows, columns))]
        final_test_data = test_data[test_data.columns.difference(cropping(rows, columns))]
        print 'Using cropped data'
        data_labels = cropped_training_data['label']
        data_features = cropped_training_data[final_test_data.columns.values]
    else:
        print 'Using non-cropped data'
        final_test_data = test_data
        data_labels = training_data['label']
        data_features = training_data[final_test_data.columns.values]
    return data_features, data_labels, final_test_data