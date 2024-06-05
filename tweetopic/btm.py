"""Module containing sklearn compatible Biterm Topic Model."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as spr
import sklearn
from numpy.typing import ArrayLike

from tweetopic._btm import (compute_biterm_set, corpus_unique_biterms,
                            fit_model, predict_docs)
from tweetopic._doc import init_doc_words
from tweetopic.exceptions import NotFittedException
from tweetopic.utils import set_numba_seed


class BTM(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """Implementation of the Biterm Topic Model with Gibbs Sampling
    solver.

    Parameters
    ----------
    n_components: int
        Number of topics in the model.
    n_iterations: int, default 100
        Number of iterations furing fitting.
    alpha: float, default 6.0
        Dirichlet prior for topic distribution.
    beta: float, default 0.1
        Dirichlet prior for topic-word distribution.

    Attributes
    ----------
    components_: array of shape (n_components, n_vocab)
        Conditional probabilities of all terms given a topic.
    topic_distribution: array of shape (n_components,)
        Prior probability of each topic.
    n_features_in_: int
        Number of total vocabulary items seen during fitting.
    random_state: int, default None
        Random seed to use for reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        n_iterations: int = 100,
        alpha: float = 6.0,
        beta: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        # Not none for typing reasons
        self.components_ = np.array(0)
        self.topic_distribution = None
        self.n_features_in_ = 0

    @property
    def _fitted(self) -> bool:
        """Property describing whether the model is fitted."""
        return self.topic_distribution is not None

    def _check_fitted(self) -> None:
        """Raise exception if the model is not fitted."""
        if not self._fitted:
            raise NotFittedException

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
            "n_iterations": self.n_iterations,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def set_params(self, **params) -> BTM:
        """Set parameters for this estimator.

        Returns
        -------
        BTM
            Estimator instance

        Note
        ----
        Exists for sklearn compatibility.
        """
        for param, value in params.items():
            self.__setattr__(param, value)
        return self

    def fit(self, X: Union[spr.spmatrix, ArrayLike], y: None = None):
        """Fits the model using Gibbs Sampling. Detailed description of the
        algorithm in Yan et al. (2013).

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            BOW matrix of corpus.
        y: None
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        BTM
            The fitted model.

        Note
        ----
        fit() works in-place too, the fitted model is returned for convenience.
        """
        if self.random_state is not None:
            set_numba_seed(self.random_state)
        # Converting X into sparse array if it isn't one already.
        X = spr.csr_matrix(X)
        _, self.n_features_in_ = X.shape
        # Calculating the number of nonzero elements for each row
        # using the internal properties of CSR matrices.
        max_unique_words = np.max(np.diff(X.indptr))
        print("Extracting biterms.")
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=max_unique_words,
        )
        biterms = corpus_unique_biterms(doc_unique_words, doc_unique_word_counts)
        biterm_set = compute_biterm_set(biterms)
        self.topic_distribution, self.components_ = fit_model(
            n_iter=self.n_iterations,
            alpha=self.alpha,
            beta=self.beta,
            n_components=self.n_components,
            n_vocab=self.n_features_in_,
            biterms=biterm_set,
        )
        return self

    # TODO: Something goes terribly wrong here, fix this

    def transform(self, X: Union[spr.spmatrix, ArrayLike]) -> np.ndarray:
        """Predicts probabilities for each document belonging to each
        topic.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        array of shape (n_samples, n_components)
            Probabilities for each document belonging to each cluster.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised
        """
        self._check_fitted()
        # Converting X into sparse array if it isn't one already.
        X = spr.csr_matrix(X)
        # n_samples, _ = X.shape
        sample_max_unique_words = np.max(np.diff(X.indptr))
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=sample_max_unique_words,
        )
        return predict_docs(
            topic_distribution=self.topic_distribution,  # type: ignore
            topic_word_distribution=self.components_,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
        )

    def fit_transform(
        self,
        X: Union[spr.spmatrix, ArrayLike],
        y: None = None,
    ) -> np.ndarray:
        """Fits the model, then transforms the given data.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        y: None
            Ignored, sklearn compatibility.

        Returns
        -------
        array of shape (n_samples, n_components)
            Probabilities for each document belonging to each cluster.
        """
        return self.fit(X).transform(X)
