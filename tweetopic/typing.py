"""Module for typing topic models and vectorizers."""
from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Union

import numpy as np
import scipy.sparse as spr
from numpy.typing import ArrayLike


class Vectorizer(Protocol):
    """Vectorizer protocol for static typing."""

    def fit(self, raw_documents: Iterable[str]) -> Vectorizer:  # type: ignore
        """Fits vectorizer on a stream of documents.

        Parameters
        ----------
        raw_documents: iterable of str
            Stream of text documents.

        Returns
        -------
        Vectorizer
            Fitted vectorizer
        """
        pass

    def transform(self, raw_documents: Iterable[str]) -> Union[np.ndarray, spr.spmatrix]:  # type: ignore
        """Transforms documents into vector embeddings.

        Parameters
        ----------
        raw_documents: iterable of str
            Stream of text documents.

        Returns
        -------
        array or sparse matrix of shape (n_documents, n_vocab)
            Document embeddings.
        """
        pass

    def fit_transform(
        self,
        raw_documents: Iterable[str],
    ) -> Union[np.ndarray, spr.spmatrix]:  # type: ignore
        """Fits vectorizer and transforms documents into vector embeddings.

        Parameters
        ----------
        raw_documents: iterable of str
            Stream of text documents.

        Returns
        -------
        array or sparse matrix of shape (n_documents, n_vocab)
            Document embeddings.
        """
        pass

    def get_feature_names_out(self) -> Union[Mapping[int, str], np.ndarray]:  # type: ignore
        """Extracts feature names from the vectorizer.

        Returns
        -------
        mapping of int to str or ndarray of string objects
            Maps term indices to feature names
        """
        pass


class TopicModel(Protocol):
    """Protocol for sklearn compatible topic models for static typing.

    Attributes
    ----------
    n_components: int
        Number of topics in the model
    components_: array-like or sparse matrix of shape (n_components, n_vocab)
        Topic-term distribution of each topic.
    """

    n_components: int
    components_: np.ndarray

    def fit(self, X: Union[ArrayLike, spr.spmatrix], y: None = None) -> TopicModel:  # type: ignore
        """Fits topic model.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_documents, n_vocab)
            Document-term matrix.
        y: Any
            Ignored, exists for compatibility.

        Returns
        -------
        TopicModel
            The fitted topic model.
        """
        pass

    def transform(
        self,
        X: Union[ArrayLike, spr.spmatrix],
    ) -> Union[np.ndarray, spr.spmatrix]:  # type: ignore
        """Transforms document embeddings into topic distributions.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_documents, n_vocab)
            Document-term matrix.

        Returns
        -------
        array or sparse matrix of shape (n_documents, n_components)
            Document-topic distribution.
        """
        pass

    def fit_transform(
        self,
        X: Union[ArrayLike, spr.spmatrix],
        y: None = None,
    ) -> Union[np.ndarray, spr.spmatrix]:  # type: ignore
        """Fits the topic model and transforms document embeddings into topic
        distributions.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_documents, n_vocab)
            Document-term matrix.
        y: Any
            Ignored, sklearn compatibility.

        Returns
        -------
        array or sparse matrix of shape (n_documents, n_components)
            Document-topic distribution.
        """
        pass
