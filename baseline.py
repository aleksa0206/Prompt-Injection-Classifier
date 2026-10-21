"""TF-IDF + logistic regression: the number the transformer has to beat."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data import report

train, test = pd.read_parquet("data/train.parquet"), pd.read_parquet("data/test.parquet")
clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
clf.fit(train.text, train.label)
report(test, clf.predict(test.text))
