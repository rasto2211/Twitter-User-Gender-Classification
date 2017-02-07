import re
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


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


def report_results(model, y, X, class_names, data_set_name):
    print(data_set_name)
    print(classification_report(y, model.predict(X), target_names=class_names))
    print("Accuracy: {}".format(accuracy_score(y, model.predict(X))))
    print("==================================================================")
    print()
