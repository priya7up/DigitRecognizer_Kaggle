Digit Recognizer

This project attempts to classify handwritten digits from MNIST data. The raw data is from Kaggle (https://www.kaggle.com/c/digit-recognizer). The details on the data and how the images of the digits are represented can be read on the Kaggle link (due to space limitations, the training and test data files have not been included here but can be downloaded from Kaggle directly).

This is one of the first Kaggle projects I attempted (after the official competition was closed though). There were several lessons that I learned from solving this problem and I will share them below. Please read the details of the competition and the file formats on the Kaggle page in order for this work to make any sense.

First an explanation of the various files needed for this project.

1. train.csv - raw training data provided by Kaggle. This file is not uploaded in this repository due to its size but is needed to be able to run the python scripts. Please download it from here: https://www.kaggle.com/c/digit-recognizer/data). 

2. test.csv - test data provided by Kaggle to use for submission. This file is not uploaded in this repository due to its size but is needed to be able to run the python scripts. Please download it from here: https://www.kaggle.com/c/digit-recognizer/data).

3. shift_image.py - Python program that processes the raw images to create more training data. Kaggle provided us with 42,000 images of various digits ranging from 0 to 9 each with 28 X 28 pixels. In order to increase the training data, this program shifts the images by one column to the left, thus creating 42,000 ‘new’ images and one column to the right, creating another 42,000 ‘new’ images and saves these 126,000 images in a new file called processed_data.csv 

4. processed_train.csv - see #3 above. This is a >200 MB file and GitHub does not allow uploading such large files so I have not made it available in the project directory. 

5. image_cropped.py - crops the image before fitting a machine learning model to the data.

6. rf_classifer.py - this file contains the machine learning algorithm used to fit the data and predict the labels on the test data.

7. digit_recognizer.csv - final results of the analysis in the format required for submission to Kaggle

I have created three python files with small amounts of code in each to modularize the codebase. You will need to run the data_processing.py file to create the csv containing the processed data. Then you can run rf_classifier.py to fit the random forest classifer to the data. Depending on whether you chose to set the 'cropped' flag or not, the image_cropped.py is called internally by the code.

There are two ways of preprocessing the data that I have explored for this project:

a. Using the random forest classifier on the raw images resulted in 96% accuracy in identifying the handwritten digits but I wanted to do better. Cropping the images improved the results some . There was a good lesson to learn here - the edges did not contain that much information and were probably acting as noise and confusing the algorithm. So the algorithm worked better when we remove this ‘noise’. In order to use cropped data, set the variable cropped = True

b. Although 42000 images is not a small amount, a larger training dataset would certainly improve results. There are many ways in which we can increase the training data size in this project. We can shift the image of the digits to the left, right, up and down. We can rotate the image or we can add random noise to randomly chosen pixels. I implemented the image shifting in the data_processing.py file. The images are simply shifted by one column to the left and to the right by shifting the pixel values respectively while looping over the last pixel value. 

Combining the image shifting process with the cropping feature improved the final result by 1%.

We can also tweak the input parameters of the random forest classifier such as the n_estimators to get more optimal results. Increasing the n_estimators from 500 to 1500 improved the results marginally. One thing to keep in mind with this project is that without any data processing, the random forest classifier already gives 96% accurate results. Hence, any improvement in predicting the results will be incremental.

The choice to use random forest algorithm (https://en.wikipedia.org/wiki/Random_forest) for this dataset was not a particularly calculated decision. The random forest classifier gives 97% accurate results (based on submission to Kaggle) meaning that it is able to guess 97% of the digits in the test data correctly. Other algorithms may perform better but the purpose of this project is not to find the exact best algorithm but to teach myself some machine learning skills. 
