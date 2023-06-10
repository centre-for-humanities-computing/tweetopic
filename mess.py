# %load_ext autoreload
# %autoreload 2
# %autoindent off

import random
from functools import partial

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
import scipy.sparse as spr
import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score
from tqdm import trange

from tweetopic._doc import init_doc_words
from tweetopic.bayesian.dmm import (BayesianDMM, posterior_predictive,
                                    predict_doc, sparse_multinomial_logpdf,
                                    symmetric_dirichlet_logpdf,
                                    symmetric_dirichlet_multinomial_logpdf)
from tweetopic.bayesian.sampling import batch_data, sample_nuts
from tweetopic.func import spread

alpha = 0.2
n_features = 10
n_docs = 1000

doc_lengths = np.random.randint(10, 100, size=n_docs)
components = stats.dirichlet.rvs(alpha=np.full(n_features, alpha))
X = np.stack([stats.multinomial.rvs(n, components[0]) for n in doc_lengths])
X = spr.csr_matrix(X)
X = X[X.getnnz(1) > 0]
n_documents, n_features_in_ = X.shape
max_unique_words = np.max(np.diff(X.indptr))
doc_unique_words, doc_unique_word_counts = init_doc_words(
    X.tolil(),
    max_unique_words=max_unique_words,
)
data = dict(
    doc_unique_words=doc_unique_words,
    doc_unique_word_counts=doc_unique_word_counts,
)


def transform(component):
    component = jnp.square(component)
    component = component / jnp.sum(component)
    return component


def logprior_fn(params):
    component = transform(params["component"])
    return symmetric_dirichlet_logpdf(component, alpha=alpha)


def loglikelihood_fn(params, data):
    doc_likelihood = jax.vmap(
        partial(sparse_multinomial_logpdf, component=params["component"])
    )
    return jnp.sum(
        doc_likelihood(
            unique_words=data["doc_unique_words"],
            unique_word_counts=data["doc_unique_word_counts"],
        )
    )


logdensity_fn(position)

logdensity_fn = lambda params: logprior_fn(params) + loglikelihood_fn(
    params, data
)
grad_estimator = blackjax.sgmcmc.gradients.grad_estimator(
    logprior_fn, loglikelihood_fn, data_size=n_documents
)
rng_key = jax.random.PRNGKey(0)
batch_key, warmup_key, sampling_key = jax.random.split(rng_key, 3)
batch_idx = batch_data(batch_key, batch_size=64, data_size=n_documents)
batches = (
    dict(
        doc_unique_words=doc_unique_words[idx],
        doc_unique_word_counts=doc_unique_word_counts[idx],
    )
    for idx in batch_idx
)
position = dict(
    component=jnp.array(
        transform(stats.dirichlet.mean(alpha=np.full(n_features, alpha)))
    )
)

samples, states = sample_nuts(position, logdensity_fn)


rng_key = jax.random.PRNGKey(0)
n_samples = 4000
sghmc = blackjax.sgld(grad_estimator)  # , num_integration_steps=10)
states = []
step_size = 1e-8
samples = []
for i in trange(n_samples, desc="Sampling"):
    _, rng_key = jax.random.split(rng_key)
    minibatch = next(batches)
    position = jax.jit(sghmc)(rng_key, position, minibatch, step_size)
    samples.append(position)

densities = [jax.jit(logdensity_fn)(sample) for sample in samples]
component_trace = jnp.stack([sample["component"] for sample in samples])
component_trace = jax.vmap(transform)(component_trace)
px.line(component_trace).show()

for i, density in enumerate(densities):
    if np.array(density) != -np.inf:
        print(f"{i}: {density}")


px.line(densities).show()
