# Sentiment-Analysis

In this project a Linear support vector classifier aims to predict whether a  sentences convey a positive or a negative sentiment. 

## Getting Started

install the python libraries mentioned in requirements.txt with python 3.x (prefer 3.7)

### Data

The data consists of 7618 negative and positive reviews in two separate file 'data/1.txt' and 'data/2.txt'. You can examine the raw reviews in `data/...`.

### Selecting the best model

Different models have been tested out and best performing has been selected for further development.

Tested models include:

    Support Vector
    Random Forest
    Logistic Regression
    Decision Tree
    MultinomialNB
    keras-CNN

The final model is LinearSVC of support Vector.

### Start the Training of The Model

To start the training and prediction of the model run the python script `sentiment_SVC.py`

During training we can observe the accuracy of 95%.

## Result

The approach taken can be found in 'report.txt'.

The consolidated result of the final model can be found in 'support_vector/data/result.txt'
