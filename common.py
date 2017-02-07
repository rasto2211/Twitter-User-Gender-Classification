import re
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


DATA_DIR = "data"
DATA_FILE = "{}/gender-classifier-DFE-791531.csv".format(DATA_DIR)


def load_data():
    return pd.read_csv(DATA_FILE, encoding='latin1')


def select_rows(df):
    return df[df["gender"].isin(["male", "female"]) &
              (df["gender:confidence"] > 0.99)].index.tolist()


def split_data(rows):
    n_samples = len(rows)
    random.shuffle(rows)
    test_size = round(n_samples * 0.4)
    test_rows = rows[:test_size]
    train_rows = rows[test_size:]

    return (train_rows, test_rows)


def encode_class_labels(train_rows, test_rows, df):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df.ix[train_rows, "gender"])
    y_test = encoder.transform(df.ix[test_rows, "gender"])

    return (y_train, y_test, encoder.classes_)


def normalize_text(text):
    # Remove non-ASCII chars.
    text = re.sub('[^\x00-\x7F]+', ' ', text)

    # Remove URLs
    text = re.sub('https?:\/\/.*[\r\n]*', ' ', text)

    # Remove special chars.
    text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '', text)

    # Remove double spaces.
    text = re.sub('\s+', ' ', text)

    return text


def load_data_split():
    train = pd.read_csv("{}/train_rows.txt".format(DATA_DIR)).row_number.tolist()
    test = pd.read_csv("{}/test_rows.txt".format(DATA_DIR)).row_number.tolist()

    return (train, test)


def print_results(y_true, y, X, data_set_name, class_names):
    print(data_set_name)
    print(classification_report(y, y_true, target_names=class_names))
    print("Accuracy: {}".format(accuracy_score(y, y_true)))
    print("==================================================================")
    print()


def report_results(grid_search, y_train, X_train, y_test, X_test, class_names):
    print("Best params: ", grid_search.best_params_)
    print_results(grid_search.predict(X_train), y_train, X_train, "Train", class_names)
    print_results(grid_search.predict(X_test), y_test, X_test, "Train", class_names)


def compute_text_feats(vectorizer, rows, df):
    return vectorizer.transform(df.ix[rows, "text_norm"])


def compute_text_desc_feats(vectorizer, rows, df):
    train_text = df.ix[rows, :]["text_norm"]
    train_desc = df.ix[rows, :]["description_norm"]

    return vectorizer.transform(train_text.str.cat(train_desc, sep=' '))


def extract_feats_from_text(df, train_rows, test_rows):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df.ix[train_rows, :]["text_norm"])

    X_train = compute_text_feats(vectorizer, train_rows, df)
    X_test = compute_text_feats(vectorizer, test_rows, df)

    return (X_train, X_test)


def extract_feats_from_text_and_desc(df, train_rows, test_rows):
    vectorizer = CountVectorizer()
    train_text = df.ix[train_rows, :]["text_norm"]
    train_desc = df.ix[train_rows, :]["description_norm"]
    vectorizer = vectorizer.fit(train_text.str.cat(train_desc, sep=' '))

    X_train = compute_text_desc_feats(vectorizer, train_rows, df)
    X_test = compute_text_desc_feats(vectorizer, test_rows, df)

    return (X_train, X_test)
