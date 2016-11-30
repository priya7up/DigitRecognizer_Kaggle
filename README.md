This project attempts to classify handwritten digits from MNIST data. The raw data is from Kaggle (https://www.kaggle.com/c/digit-recognizer). The details on the data and how the images of the digits are represented can be read on the Kaggle link.

This was one of the first Kaggle projects I attempted (after the official competition was closed though). There were several lessons that I learned from solving this problem and I will share them below.

First an explanation of the various files needed for this project.
1. train.csv - raw training data provided by Kaggle ( please read about the way the images are presented here: https://www.kaggle.com/c/digit-recognizer/data)
2. test.csv - test data provided by Kaggle to use for submission
3. data_processing.py - Python program that processes the raw images to create more training data. Kaggle provided us with 42000 images of various digits ranging from 0 to 9 each with 28X28 pixels. In order to increase the training data, this program shifts the images by one column to the left and one column to the right and saves the 126000 images in a new file called processed_data.csv
4. processed_data.csv - see #3 above. This is a >200 MB file and GitHub does not allow uploading such large files so I have not made it available in the project directory
5. image_cropped.py - crops the first two and last two rows and columns if user wants to crop the image before fitting
6. rf_classifer.py - this file contains the meat of the machine learning algorithm.
7. digit_recognizer.csv - final results of the analysis in the format for submission to Kaggle

I have created three python files with very little code in each just for clarity. This is not the most efficient code, just code that works and produces reasonable results.
There are two ways of processing the data that I have explored for this project.
a. Using the random forest classifier on the raw images resulted in reasonable accuracy in identifying the handwritten digits but I wanted to do better. Cropping the images from the edges evenly by about 25% improved the results . There was a good lesson to learn here - the edges did not contain that much information and was probably acting as noise and confusing the algorithm. So the algorithm worked better when we remove this ‘noise’. I use this concept in future projects as well (for example, the SF crime project). In order use cropped data, set the variable cropped = True
b. Although 42000 images is not a small amount, a larger training dataset would certainly improve results. There are many ways in which we can increase the training data size in this project. We can shift the image of the digits to the left, right, up and down. We can rotate the image or we can add random noise to randomly chosen pixels. I implemented the image shifting in the data_processing.py file. The images are simple shifted by one column to the left and to the right by shifting the pixel values respectively and looping over the last pixel value. 
Combining the image shifting process with the cropping feature improved the final result by several points.

We can also play around with the input parameters of the random forest classifier such as the n_estimators to get more optimal results. Increasing the n_estimators from 500 to 1500 improved the results only marginally but there could be other optimizations that can be performed. 

The choice to use random forest algorithm for this dataset was not a particularly calculated decision. The random forest classifier gives 97% accurate results (based on submission to Kaggle) meaning that it is able to guess 97% of the digits in the test data correctly. Other algorithms such as SVM could perform better but purpose of this project as not to find the exact best algorithm but to teach myself some machine learning skills.
