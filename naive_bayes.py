from common import load_data
from common import load_data_split
from common import encode_class_labels
from common import report_results
from common import extract_feats_from_text
from common import extract_feats_from_text_and_desc

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


JOBS = 4
PARAMS = [{'alpha': [8, 4, 2, 1, 0.5, 0.25, 0.1, 0.07, 0.05, 0.03, 0.01, 0.001]}]


df = load_data()
train_rows, test_rows = load_data_split()

y_train, y_test, class_names = \
    encode_class_labels(train_rows, test_rows, df)

print("Features only from Text")

X_train, X_test = extract_feats_from_text(df, train_rows, test_rows)

grid_search = GridSearchCV(MultinomialNB(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)
report_results(grid_search, y_train, X_train, y_test, X_test, class_names)

print("Features from tweet text and description")

X_train, X_test = extract_feats_from_text_and_desc(df, train_rows, test_rows)

grid_search = GridSearchCV(MultinomialNB(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)
report_results(grid_search, y_train, X_train, y_test, X_test, class_names)
