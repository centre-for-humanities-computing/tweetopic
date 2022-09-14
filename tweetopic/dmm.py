"""Module containing a fully sklearn compatible Dirichlet Mixture Model."""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as spr
import sklearn
from numpy.typing import ArrayLike

from tweetopic._doc import init_doc_words
from tweetopic._mgp import fit_model, init_clusters
from tweetopic._prob import predict_doc
from tweetopic.exceptions import NotFittedException


class DMM(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """Implementation of the Dirichlet Mixture Model with Gibbs Sampling
    solver. The class aims to achieve full compatibility with sklearn.

    Parameters
    ----------
    n_components: int, default 8
        Number of mixture components in the model.
    n_iterations: int, default 30
        Number of iterations during fitting.
        If the model converges earlier, fitting will stop.
    alpha: float, default 0.1
        Willingness of a document joining an empty cluster.
    beta: float, default 0.1
        Willingness to join clusters, where the terms in the document
        are not present.

    Attributes
    ----------
    components_: array of shape (n_components, n_vocab)
        Describes all components of the topic distribution.
        Contains the amount each word has been assigned to each component
        during fitting.
    cluster_doc_count: array of shape (n_components,)
        Array containing how many documents there are in each cluster.
    n_features_in_: int
        Number of total vocabulary items seen during fitting.
    n_documents: int
        Total number of documents seen during fitting.
    max_unique_words: int
        Maximum number of unique words in a document seen during fitting.
    """

    def __init__(
        self,
        n_components: int = 8,
        n_iterations: int = 30,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        # Not none for typing reasons
        self.components_ = np.array(0)
        self.cluster_doc_count = None
        self.cluster_word_count = None
        self.n_features_in_ = 0
        self.n_documents = 0
        self.max_unique_words = 0

    @property
    def _fitted(self) -> bool:
        """Property describing whether the model is fitted."""
        # If the number of documents seen is more than 0
        # It can be assumed that the model is fitted.
        return bool(self.n_documents)

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

    def set_params(self, **params) -> DMM:
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

    def fit(self, X: Union[spr.spmatrix, ArrayLike], y: None = None):
        """Fits the model using Gibbs Sampling. Detailed description of the
        algorithm in Yin and Wang (2014).

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            BOW matrix of corpus.
        y: None
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        DMM
            The fitted model.

        Note
        ----
        fit() works in-place too, the fitted model is returned for convenience.
        """
        # Converting X into sparse array if it isn't one already.
        X = spr.csr_matrix(X)
        self.n_documents, self.n_features_in_ = X.shape
        # Calculating the number of nonzero elements for each row
        # using the internal properties of CSR matrices.
        self.max_unique_words = np.max(np.diff(X.indptr))
        print("Calculating unique words.")
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=self.max_unique_words,
        )
        print("Initialising mixture components")
        initial_clusters = np.random.multinomial(
            1,
            np.ones(self.n_components) / self.n_components,
            size=self.n_documents,
        )
        doc_clusters = np.argmax(initial_clusters, axis=1)
        self.cluster_doc_count = np.zeros(self.n_components)
        self.components_ = np.zeros((self.n_components, self.n_features_in_))
        self.cluster_word_count = np.zeros(self.n_components)
        init_clusters(
            cluster_word_distribution=self.components_,
            cluster_word_count=self.cluster_word_count,
            cluster_doc_count=self.cluster_doc_count,
            doc_clusters=doc_clusters,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            max_unique_words=self.max_unique_words,
        )
        print("Fitting model")
        fit_model(
            n_iter=self.n_iterations,
            alpha=self.alpha,
            beta=self.beta,
            n_clusters=self.n_components,
            n_vocab=self.n_features_in_,
            n_docs=self.n_documents,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            doc_clusters=doc_clusters,
            cluster_doc_count=self.cluster_doc_count,
            cluster_word_count=self.cluster_word_count,
            cluster_word_distribution=self.components_,
            max_unique_words=self.max_unique_words,
        )
        return self

    def transform(self, X: Union[spr.spmatrix, ArrayLike]) -> np.ndarray:
        """Predicts probabilities for each document belonging to each
        component.

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
        sample_max_unique_words = np.max(np.diff(X.indptr))
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=sample_max_unique_words,
        )
        doc_words_count = np.sum(doc_unique_word_counts, axis=1)
        n_docs = X.shape[0]
        predictions = []
        for i_doc in range(n_docs):
            pred = np.zeros(self.n_components)
            predict_doc(
                probabilities=pred,
                i_document=i_doc,
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
                n_words=doc_words_count[i_doc],
                alpha=self.alpha,
                beta=self.beta,
                n_clusters=self.n_components,
                n_vocab=self.n_features_in_,
                n_docs=n_docs,
                cluster_doc_count=self.cluster_doc_count,  # type: ignore
                cluster_word_count=self.cluster_word_count,  # type: ignore
                cluster_word_distribution=self.components_,  # type: ignore
                max_unique_words=sample_max_unique_words,
            )
            predictions.append(pred)
        return np.stack(predictions)

    def predict_proba(self, X: Union[spr.spmatrix, ArrayLike]) -> np.ndarray:
        """Alias of :meth:`~tweetopic.dmm.DMM.transform` .

        Mainly exists for compatibility with density estimators in
        sklearn.
        """
        return self.transform(X)

    def predict(self, X: Union[spr.spmatrix, ArrayLike]) -> np.ndarray:
        """Predicts cluster labels for a set of documents. Mainly exists for
        compatibility with density estimators in sklearn.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        array of shape (n_samples,)
            Cluster label for each document.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised
        """
        return np.argmax(self.transform(X), axis=1)

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
