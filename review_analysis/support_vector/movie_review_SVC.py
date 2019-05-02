from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


def get_training_data():
    data1 = pd.read_csv("data/1.txt", sep="\t", header=None)
    data2 = pd.read_csv("data/2.txt", sep="\t", header=None)
    data3 = data2[[1, 0]]
    data3.columns = range(data3.shape[1])
    result = pd.concat([data1, data3], ignore_index=True)
    result.columns = ["review", "sentiment"]
    result = result.sample(frac=1).reset_index(drop=True)
    return result


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
    data = get_training_data()
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
    print(data_to_predict.head())
    data_to_predict.to_csv('data/result.txt', sep='\t')


if __name__ == "__main__":
    main()
