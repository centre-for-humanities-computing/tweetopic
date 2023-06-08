# %load_ext autoreload
# %autoreload 2
# %autoindent off

import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
import scipy.sparse as spr
import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score

from tweetopic._doc import init_doc_words
from tweetopic.bayesian.dmm import (
    BayesianDMM,
    posterior_predictive,
    predict_doc,
    sparse_multinomial_logpdf,
)
from tweetopic.sampling import (
    sample_meanfield_vi,
    sample_nuts,
    sample_pathfinder,
    sample_sgld,
)

texts = [line for line in open("processed_sample.txt")]

vectorizer = CountVectorizer(max_features=100, max_df=0.3, min_df=10)
X = vectorizer.fit_transform(random.sample(texts, 10_000))

model = BayesianDMM(
    n_components=5,
    alpha=1.0,
    beta=1.0,
    sampler=partial(sample_sgld, n_samples=2000),
)
model.fit(X)

X = X[X.getnnz(1) > 0]
X = spr.csr_matrix(X)
max_unique_words = np.max(np.diff(X.indptr))
doc_unique_words, doc_unique_word_counts = init_doc_words(
    X.tolil(),
    max_unique_words=max_unique_words,
)

components = np.array([sample["components"] for sample in model.samples])
weights = np.array([sample["weights"] for sample in model.samples])

pred = posterior_predictive(
    doc_unique_words, doc_unique_word_counts, components, weights
)

px.box(pred[4]).show()

pred[0]

px.line(weights).show()

X.shape

predict_doc()

try:
    predict_one_doc(doc=docs[0], samples=np.array(model.samples[:2]))
except Exception:
    print("oopsie")
