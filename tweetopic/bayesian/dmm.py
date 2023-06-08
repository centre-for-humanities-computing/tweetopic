"""JAX implementation of probability densities and parameter initialization
for the Dirichlet Multinomial Mixture Model."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as spr
import scipy.stats
import sklearn
from sklearn.exceptions import NotFittedError
from tqdm import tqdm, trange

from tweetopic._doc import init_doc_words
from tweetopic.bayesian.sampling import Sampler, sample_sgld
from tweetopic.func import spread


def symmetric_dirichlet_multinomial_mean(alpha: float, n: int, K: int):
    """Returns mean of a symmetric dirichlet multinomial."""
    return np.full(K, n * alpha / K * alpha)


def init_parameters(
    n_docs: int, n_vocab: int, n_components: int, alpha: float, beta: float
) -> dict:
    """Initializes the parameters of the dmm to the mean of the prior."""
    return dict(
        weights=symmetric_dirichlet_multinomial_mean(
            alpha, n_docs, n_components
        ),
        components=np.broadcast_to(
            scipy.stats.dirichlet.mean(np.full(n_vocab, beta)),
            (n_components, n_vocab),
        ),
    )


def sparse_multinomial_logpdf(
    component,
    unique_words,
    unique_word_counts,
):
    """Calculates joint multinomial probability of a sparse representation"""
    unique_word_counts = jnp.float64(unique_word_counts)
    n_words = jnp.sum(unique_word_counts)
    n_factorial = jax.lax.lgamma(n_words + 1)
    word_count_factorial = jax.lax.lgamma(unique_word_counts + 1)
    word_count_factorial = jnp.where(
        unique_word_counts != 0, word_count_factorial, 0
    )
    denominator = jnp.sum(word_count_factorial)
    probs = component[unique_words]
    numerator_terms = probs * unique_word_counts
    numerator_terms = jnp.where(unique_word_counts != 0, numerator_terms, 0)
    numerator = jnp.sum(numerator_terms)
    return n_factorial + numerator - denominator


def symmetric_dirichlet_logpdf(x, alpha):
    """Logdensity of a symmetric Dirichlet."""
    K = x.shape[0]
    return (
        jax.lax.lgamma(alpha * K)
        - K * jax.lax.lgamma(alpha)
        + (alpha - 1) * jnp.sum(jnp.log(x))
    )


def symmetric_dirichlet_multinomial_logpdf(x, n, alpha):
    """Logdensity of a symmetric Dirichlet Multinomial."""
    K = x.shape[0]
    return (
        jax.lax.lgamma(K * alpha)
        + jax.lax.lgamma(n + 1)
        - jax.lax.lgamma(n + K * alpha)
        - K * jax.lax.lgamma(alpha)
        + jnp.sum(jax.lax.lgamma(x + alpha) - jax.lax.lgamma(x + 1))
    )


def predict_doc(components, weights, unique_words, unique_word_counts):
    """Predicts likelihood of a document belonging to
    each cluster based on given parameters."""
    component_logpdf = partial(
        sparse_multinomial_logpdf,
        unique_words=unique_words,
        unique_word_counts=unique_word_counts,
    )
    component_logprobs = jax.lax.map(component_logpdf, components) + jnp.log(
        weights
    )
    norm_probs = jnp.exp(
        component_logprobs - jax.scipy.special.logsumexp(component_logprobs)
    )
    return norm_probs


def predict_one(unique_words, unique_word_counts, components, weights):
    return jax.vmap(
        partial(
            predict_doc,
            unique_words=unique_words,
            unique_word_counts=unique_word_counts,
        )
    )(components, weights)


def posterior_predictive(
    doc_unique_words, doc_unique_word_counts, components, weights
):
    """Predicts probability of a document belonging to each component
    for all posterior samples.
    """
    predict_all = jax.vmap(
        partial(predict_one, components=components, weights=weights)
    )
    return predict_all(doc_unique_words, doc_unique_word_counts)


def dmm_loglikelihood(
    components, weights, doc_unique_words, doc_unique_word_counts, alpha, beta
):
    docs = jnp.stack((doc_unique_words, doc_unique_word_counts), axis=1)

    def doc_likelihood(doc):
        unique_words, unique_word_counts = doc
        component_logpdf = partial(
            sparse_multinomial_logpdf,
            unique_words=unique_words,
            unique_word_counts=unique_word_counts,
        )
        component_logprobs = jax.lax.map(
            component_logpdf, components
        ) + jnp.log(weights)
        return jax.scipy.special.logsumexp(component_logprobs)

    loglikelihood = jnp.sum(jax.lax.map(doc_likelihood, docs))
    return loglikelihood


def dmm_logprior(components, weights, alpha, beta, n_docs):
    components_prior = jnp.sum(
        jax.lax.map(
            partial(symmetric_dirichlet_logpdf, alpha=alpha), components
        )
    )
    weights_prior = symmetric_dirichlet_multinomial_logpdf(
        weights, n=jnp.float64(n_docs), alpha=beta
    )
    return components_prior + weights_prior


def dmm_logpdf(
    components, weights, doc_unique_words, doc_unique_word_counts, alpha, beta
):
    """Calculates logdensity of the DMM at a given point in parameter space."""
    n_docs = doc_unique_words.shape[0]
    loglikelihood = dmm_loglikelihood(
        components,
        weights,
        doc_unique_words,
        doc_unique_word_counts,
        alpha,
        beta,
    )
    logprior = dmm_logprior(components, weights, alpha, beta, n_docs)
    return logprior + loglikelihood


class BayesianDMM(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """Fully Bayesian Dirichlet Multinomial Mixture Model."""

    def __init__(
        self,
        n_components: int,
        sampler: Sampler = sample_sgld,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.sampler = sampler

    def get_params(self, deep: bool = False) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: bool, default False
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        Note
        ----
        Exists for sklearn compatibility.
        """
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def set_params(self, **params):
        """Set parameters for this estimator.

        Returns
        -------
        DMM
            Estimator instance

        Note
        ----
        Exists for sklearn compatibility.
        """
        for param, value in params:
            self.__setattr__(param, value)
        return self

    def fit(self, X, y=None):
        # Filtering out empty documents
        X = X[X.getnnz(1) > 0]
        # Converting X into sparse array if it isn't one already.
        X = spr.csr_matrix(X)
        self.n_documents, self.n_features_in_ = X.shape
        # Calculating the number of nonzero elements for each row
        # using the internal properties of CSR matrices.
        self.max_unique_words = np.max(np.diff(X.indptr))
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=self.max_unique_words,
        )
        initial_position = init_parameters(
            n_docs=self.n_documents,
            n_components=self.n_components,
            n_vocab=self.n_features_in_,
            alpha=self.alpha,
            beta=self.beta,
        )
        logdensity_fn = partial(
            dmm_logpdf,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            alpha=self.alpha,
            beta=self.beta,
        )
        samples = self.sampler(
            initial_position,
            logdensity_fn,
            data=dict(
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
            ),
        )
        self.samples = samples
        return self

    def posterior_predictive(self, X):
        try:
            samples = self.samples
        except AttributeError:
            raise NotFittedError("The posterior has yet to be sampled.")

    def predict_proba(self, X) -> np.ndarray:
        return self.transform(X)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.transform(X), axis=1)

    def fit_transform(
        self,
        X,
        y: None = None,
    ) -> np.ndarray:
        return self.fit(X).transform(X)
