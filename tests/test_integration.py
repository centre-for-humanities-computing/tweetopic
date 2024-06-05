import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

from tweetopic import BTM, DMM

newsgroups = fetch_20newsgroups(
    subset="all",
    categories=[
        "misc.forsale",
    ],
    remove=("headers", "footers", "quotes"),
)
texts = newsgroups.data

models = [DMM(10), BTM(10)]


@pytest.mark.parametrize("model", models)
def test_fit(model):
    pipe = make_pipeline(CountVectorizer(), model)
    doc_topic_matrix = pipe.fit_transform(texts)
    assert doc_topic_matrix.shape[0] == len(texts)
