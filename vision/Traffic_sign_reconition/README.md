# Traffic Signs Recognition with Tensorflow

I'll start with a simple goal: classifiction. Given an image of a traffic sign, our model should be able to tell it's type (e.g. Stop sign, speed limit, yield sign, ...etc.). We'll work with images that are properly cropped such that the traffic sign takes most of the image.

For this project, I'm using Pythong 3.5, Tensorflow 0.11, Numpy, Sci-kit Image, and Matplotlib.

Note that I have the files of this project in the ~/traffic directory, and I'm it to /traffic directory in the Docker container. Modify this if you're using a different directory.

**Trainging Dataset**

We're using the Belgian Traffic Sign Dataset. Go to http://btsd.ethz.ch/shareddata/ and download the training and test data. There is a lot of datasets on that page, but you only need the two files listed under BelgiumTS for Classification (cropped images)":

BelgiumTSC_Training (171.3MBytes)
BelgiumTSC_Testing (76.5MBytes)
After downloading and expanding the files, your directory structure should look something like this:

/traffic/datasets/BelgiumTS/Training/
/traffic/datasets/BelgiumTS/Testing/
Each of the two directories above has 62 sub-directories named sequentially from 00000 to 00062. The directory name represents the code (or label) and the images inside the directory are examples of that label.
Parse and Load the Training Data

The Training directory contains sub-directories with sequental numerical names from 00000 to 00061. The name of the directory represents the labels from 0 to 61, and the images in each directory represent the traffic signs that belong to that label. The images are saved in the not-so-common .ppm format, but luckily, this format is supported in the skimage library.