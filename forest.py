from common import load_data
from common import load_data_split
from common import encode_class_labels
from common import report_results
from common import extract_tfidf_from_text_and_desc
from common import extract_feats_from_text_and_desc
from common import extract_tweet_count_feats

from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


JOBS = 4
PARAMS = [{'max_depth': [150, 160, 170, 180, 190, 200, 220],
           'n_estimators': [120, 140, 160, 180, 200, 220, 240, 260]}]


df = load_data()
train_rows, test_rows = load_data_split()

y_train, y_test, class_names = \
    encode_class_labels(train_rows, test_rows, df)

print("TIDF")

X_train, X_test = extract_tfidf_from_text_and_desc(df, train_rows, test_rows)

tweet_feats_train, tweet_feats_test = extract_tweet_count_feats(df, train_rows, test_rows)

# Merge tweets feats and TF-IDF.
X_train = hstack((X_train, tweet_feats_train))
X_test = hstack((X_test, tweet_feats_test))

grid_search = GridSearchCV(RandomForestClassifier(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)
report_results(grid_search, y_train, X_train, y_test, X_test, class_names)

print("Count Vectorizer")

X_train, X_test = extract_feats_from_text_and_desc(df, train_rows, test_rows)

# Merge tweets feats and TF-IDF.
X_train = hstack((X_train, tweet_feats_train))
X_test = hstack((X_test, tweet_feats_test))

grid_search = GridSearchCV(RandomForestClassifier(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                           scoring="f1")
grid_search.fit(X_train, y_train)
report_results(grid_search, y_train, X_train, y_test, X_test, class_names)
