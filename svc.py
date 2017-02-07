from common import load_data
from common import load_data_split
from common import encode_class_labels
from common import report_results
from common import extract_feats_from_text_and_desc
from common import extract_tweet_count_feats

from scipy.sparse import hstack
from sklearn.svm import  SVC
from sklearn.model_selection import GridSearchCV


JOBS = 4
PARAMS = [{'C': [4, 2, 1.5, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001],
           'kernel': ["linear", "poly", "rbf", "sigmoid"],
           'cache_size': [1000],
           'gamma': ['auto', 1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]}]


df = load_data()
train_rows, test_rows = load_data_split()

y_train, y_test, class_names = \
    encode_class_labels(train_rows, test_rows, df)

print("Features from tweet text and description")

X_train, X_test = extract_feats_from_text_and_desc(df, train_rows, test_rows)

grid_search = GridSearchCV(SVC(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)
report_results(grid_search, y_train, X_train, y_test, X_test, class_names)

print("Features from tweet text, description, retweet count, tweet count"
      " and number of favorite tweets of the user")

tweet_feats_train, tweet_feats_test = extract_tweet_count_feats(df, train_rows, test_rows)

X_train = hstack((X_train, tweet_feats_train))
X_test = hstack((X_test, tweet_feats_test))

grid_search = GridSearchCV(SVC(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)
report_results(grid_search, y_train, X_train, y_test, X_test, class_names)
