from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import re


def get_cleaned_training_data():
    data1 = pd.read_csv("data/1.txt", sep="\t", header=None)
    data2 = pd.read_csv("data/2.txt", sep="\t", header=None)
    data3 = data2[[1, 0]]
    data3.columns = range(data3.shape[1])
    result = pd.concat([data1, data3], ignore_index=True)
    result.columns = ["review", "sentiment"]
    contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                        "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                        "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                        "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                        "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                        "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                        "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                        "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                        "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                        "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                        "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                        "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                        "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                        "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                        "she's": "she is", "should've": "should have", "shouldn't": "should not",
                        "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is",
                        "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                        "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                        "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                        "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                        "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                        "we've": "we have", "weren't": "were not", "what'll": "what will",
                        "what'll've": "what will have", "what're": "what are", "what's": "what is",
                        "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                        "where's": "where is", "where've": "where have", "who'll": "who will",
                        "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                        "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have"}

    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))

    def expand_contractions(text, c_re=contraction_re):
        def replace(match):
            return contraction_dict[match.group(0)]

        return c_re.sub(replace, text.lower())

    lis1 = []
    for i in range(result.shape[0]):
        lis1.append(clean_text(expand_contractions(result["review"][i])))
    print(max(len(x) for x in lis1))
    result["review"] = np.asarray(lis1)
    result = result.sample(frac=1).reset_index(drop=True)
    return result


def clean_text(text):
    text = str(text)
    for punct in '?!.,"#$&%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        text = text.replace(punct, '')
    text = " ".join(text.split())
    return text


def train_test_split(x, y):
    split_ratio = 0.1
    split_index = int(split_ratio * len(x))
    text_train, y_train = x[:split_index], y[:split_index]
    text_test, y_test = x[split_index:], y[split_index:]
    return text_train, y_train, text_test, y_test


def tokenize(x_train, x_test):
    vect = CountVectorizer(min_df=5, ngram_range=(1, 1), stop_words='english')
    train = vect.fit(x_train).transform(x_train)
    test = vect.transform(x_test)
    return vect, train, test


def model(x_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 5, 10]}
    grid = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
    grid.fit(x_train, y_train)
    print("Best estimator: ", grid.best_estimator_)
    classifier = grid.best_estimator_
    return classifier


def main():
    data = get_cleaned_training_data()
    text = list(data.review)
    label = list(data.sentiment)
    text_train, y_train, text_test, y_test = train_test_split(text, label)
    vect, train, test = tokenize(text_train, text_test)
    clf = model(train, y_train)
    clf.predict(test)
    print("Score: {:.2f}".format(clf.score(test, y_test)))
    data_to_predict = pd.read_csv("data/3.txt", sep=",", index_col=0, header='infer')
    predict_data = data_to_predict.loc[:, "Text"]
    vec_pred = vect.transform(predict_data)
    predicted = clf.predict(vec_pred)
    data_to_predict["Prediction"] = np.asarray(predicted)
    print(data_to_predict.head(20))
    data_to_predict.to_csv('data/result.txt', sep='\t')


if __name__ == "__main__":
    main() 
