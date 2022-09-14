"""Module for typing topic models and vectorizers."""
from __future__ import annotations

from typing import Iterable, Protocol, Union

import scipy.sparse as spr
from numpy.typing import ArrayLike


class Vectorizer(Protocol):
    """Vectorizer protocol for static typing."""

    def fit(self, raw_documents: Iterable[str]) -> Vectorizer:  # type: ignore
        pass

    def transform(self, raw_documents: Iterable[str]) -> Union[ArrayLike, spr.spmatrix]:  # type: ignore
        pass

    def fit_transform(
        self, raw_documents: Iterable[str]
    ) -> Union[ArrayLike, spr.spmatrix]:  # type: ignore
        pass

    def get_feature_names_out(self) -> ArrayLike:  # type: ignore
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
    components_: Union[ArrayLike, spr.spmatrix]

    def fit(self, embeddings: Union[ArrayLike, spr.spmatrix]) -> TopicModel:  # type: ignore
        """Fits topic model.

        Parameters
        ----------
        embeddings: array-like or sparse matrix of shape (n_documents, n_vocab)
            Document-term matrix.

        Returns
        -------
        TopicModel
            The fitted topic model.
        """
        pass

    def transform(
        self, embeddings: Union[ArrayLike, spr.spmatrix]
    ) -> Union[ArrayLike, spr.spmatrix]:  # type: ignore
        """Transforms document embeddings into topic distributions.

        Parameters
        ----------
        embeddings: array-like or sparse matrix of shape (n_documents, n_vocab)
            Document-term matrix.

        Returns
        -------
        array-like or sparse matrix of shape (n_documents, n_components)
            Document-topic distribution.
        """
        pass

    def fit_transform(
        self, embeddings: Union[ArrayLike, spr.spmatrix]
    ) -> Union[ArrayLike, spr.spmatrix]:  # type: ignore
        """Fits the topic model and transforms
        document embeddings into topic distributions.

        Parameters
        ----------
        embeddings: array-like or sparse matrix of shape (n_documents, n_vocab)
            Document-term matrix.

        Returns
        -------
        array-like or sparse matrix of shape (n_documents, n_components)
            Document-topic distribution.
        """
        pass
