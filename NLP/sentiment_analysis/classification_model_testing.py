import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def get_data():
    data1 = pd.read_csv("support_vector/data/1.txt", sep="\t", header=None)
    data2 = pd.read_csv("support_vector/data/2.txt", sep="\t", header=None)
    data3 = data2[[1, 0]]
    data3.columns = range(data3.shape[1])
    result = pd.concat([data1, data3], ignore_index=True)
    result.columns = ["text", "label"]
    result = result.sample(frac=1).reset_index(drop=True)
    return result


def clean_text(text):
    text = str(text)
    for punct in '?!.,"#$&%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        text = text.replace(punct, '')
    text = " ".join(text.split())
    return text


def train_test_split_fun(df, keras=False):
    if keras:
        y = to_categorical(df['label'])
    else:
        y = df['label']
    train_df = df['text']
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    x_train_counts = count_vect.fit_transform(train_df)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test


def model_def1():
    # Randomforest
    rfc = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=200, max_depth=8, criterion='gini')
    #    rfc=SVC(kernel='rbf',decision_function_shape='ovr',max_iter=9000)
    # SVM 
    model = LinearSVC(max_iter=9000)
    return rfc, model


def model_def2():
    # logistic regression
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=9000, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)
    # Naive Bayes
    gnb = MultinomialNB()
    return model, gnb


def model_def3():
    # Desicion Tree with Gini Index 
    dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                         max_depth=5, min_samples_leaf=5)
    # Desicion Tree with Information Gain 
    dt_clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                            max_depth=5, min_samples_leaf=5)
    return dt_clf_gini, dt_clf_entropy


def train_model(x_train, y_train):
    model1, model1_2 = model_def1()
    model2, model2_1 = model_def2()
    model3, model3_1 = model_def3()
    print("different model fitting")
    model1.fit(x_train, y_train)
    model1_2.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model2_1.fit(x_train, y_train)
    model3.fit(x_train, y_train)
    model3_1.fit(x_train, y_train)
    print("all models fitted")
    # save the model to disk
    joblib.dump(model1, 'rand_forest_model.sav')
    joblib.dump(model1_2, 'SVM_model.sav')
    joblib.dump(model2, 'logistic_regression_model.sav')
    joblib.dump(model2_1, 'MultiNB_model.sav')
    joblib.dump(model3, 'decision_tree_gini_model.sav')
    joblib.dump(model3_1, 'decision_tree_entropy_model.sav')
    return print("models trained and saved")


def test_model(x_test, y_test):
    print("score testing of each model")
    load_model1 = joblib.load('rand_forest_model.sav')
    load_model2 = joblib.load('logistic_regression_model.sav')
    load_model3 = joblib.load('decision_tree_gini_model.sav')
    load_model4 = joblib.load('decision_tree_entropy_model.sav')
    load_model5 = joblib.load('SVM_model.sav')
    load_model6 = joblib.load('MultiNB_model.sav')
    # load model    
    result1 = load_model1.score(x_test, y_test)
    result2 = load_model2.score(x_test, y_test)
    result3 = load_model3.score(x_test, y_test)
    result4 = load_model4.score(x_test, y_test)
    result5 = load_model5.score(x_test, y_test)
    result6 = load_model6.score(x_test, y_test)
    print("Accuracy for Random forest on data: ", result1)
    print("Accuracy for Logis Regression on data: ", result2)
    print("Accuracy for DT gini on data: ", result3)
    print("Accuracy for DT entropy on data: ", result4)
    print("Accuracy for SVM on data: ", result5)
    print("Accuracy for gaussianNB on data: ", result6)


def main():
    data = get_data()
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
    for i in range(data.shape[0]):
        lis1.append(clean_text(expand_contractions(data["text"][i])))
    data["text"] = np.asarray(lis1)
    print(data.tail())
    x_train, x_test, y_train, y_test = train_test_split_fun(data)
    train_model(x_train, y_train)
    test_model(x_test, y_test)


if __name__ == "__main__":
    main()
