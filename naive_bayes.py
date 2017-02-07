from common import load_data
from common import load_data_split
from common import encode_class_labels
from common import normalize_text
from common import report_results

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


JOBS = 4
PARAMS = [{'alpha': [8, 4, 2, 1, 0.5, 0.25, 0.1, 0.07, 0.05, 0.03, 0.01, 0.001]}]


def compute_text_feats(vectorizer, rows, df):
    return vectorizer.transform(df.ix[rows, "text_norm"])


def compute_text_desc_feats(vectorizer, rows, df):
    train_text = df.ix[rows, :]["text_norm"]
    train_desc = df.ix[rows, :]["description_norm"]

    return vectorizer.transform(train_text.str.cat(train_desc, sep=' '))


df = load_data()
train_rows, test_rows = load_data_split()

y_train, y_test, class_names = \
    encode_class_labels(train_rows, test_rows, df)

df["text_norm"] = [normalize_text(text) for text in df["text"]]
df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

print("Features only from Text")

vectorizer = CountVectorizer()
vectorizer = vectorizer.fit(df.ix[train_rows, :]["text_norm"])

X_train = compute_text_feats(vectorizer, train_rows, df)
X_test = compute_text_feats(vectorizer, test_rows, df)

grid_search = GridSearchCV(MultinomialNB(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)

print("Best params: ", grid_search.best_params_)
report_results(grid_search, y_train, X_train, class_names, "Train")
report_results(grid_search, y_test, X_test, class_names, "Test")

print("Features from tweet text and description")

vectorizer = CountVectorizer()
train_text = df.ix[train_rows, :]["text_norm"]
train_desc = df.ix[train_rows, :]["description_norm"]
vectorizer = vectorizer.fit(train_text.str.cat(train_desc, sep=' '))

X_train = compute_text_desc_feats(vectorizer, train_rows, df)
X_test = compute_text_desc_feats(vectorizer, test_rows, df)

grid_search = GridSearchCV(MultinomialNB(), PARAMS, n_jobs=JOBS, verbose=5, cv=5,
                           scoring="f1")
grid_search.fit(X_train, y_train)

print("Best params: ", grid_search.best_params_)
report_results(grid_search, y_train, X_train, class_names, "Train")
report_results(grid_search, y_test, X_test, class_names, "Test")
