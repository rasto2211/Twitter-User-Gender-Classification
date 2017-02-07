import re
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


DATA_FILE = "data/gender-classifier-DFE-791531.csv"


def load_data():
    return pd.read_csv(DATA_FILE, encoding='latin1')


def select_rows(df):
    return df[df["gender"].isin(["male", "female"]) &
              (df["gender:confidence"] > 0.99)].index.tolist()


def split_data(rows):
    n_samples = len(rows)
    random.shuffle(rows)
    test_size = round(n_samples * 0.2)
    test_rows = rows[:test_size]
    val_rows = rows[test_size:2 * test_size]
    train_rows = rows[2 * test_size:]

    return (train_rows, val_rows, test_rows)


def encode_labels(train_rows, val_rows, test_rows, df):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df.ix[train_rows, "gender"])
    y_val = encoder.transform(df.ix[val_rows, "gender"])
    y_test = encoder.transform(df.ix[test_rows, "gender"])

    return (y_train, y_val, y_test)


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
