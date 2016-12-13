def crop(data, rows, columns):
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
    pixels_to_crop = ['pixel' + str(x) for x in output]
    return data[data.columns.difference(pixels_to_crop)]
