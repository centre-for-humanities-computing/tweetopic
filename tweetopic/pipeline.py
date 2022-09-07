"""Utilities for easier fitting, inspection and visualization of topic
models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Union

import numpy as np
import scipy.sparse as spr
from numpy.typing import ArrayLike
from sklearn.decomposition import NMF, LatentDirichletAllocation

from tweetopic.dmm import DMM


class Vectorizer(Protocol):
    """Vectorizer protocol for static typing purposes."""

    def fit(self, raw_documents) -> Vectorizer:  # type: ignore
        pass

    def transform(self, raw_documents) -> Union[ArrayLike, spr.spmatrix]:  # type: ignore
        pass

    def fit_transform(self, raw_documents) -> Union[ArrayLike, spr.spmatrix]:  # type: ignore
        pass

    def get_feature_names_out(self) -> np.ndarray:  # type: ignore
        pass


TopicModel = Union[NMF, LatentDirichletAllocation, DMM]


@dataclass
class TopicPipeline:
    """Provides a full pipeline for topic modelling with useful utilities.

    Parameters
    ----------
    vectorizer: Vectorizer
        Transformer that extracts BOW embeddings from texts
    topic_model: DMM, NMF or LatentDirichletAllocation
        Topic model to add to the pipeline.
    """

    vectorizer: Vectorizer
    topic_model: TopicModel

    def fit(self, texts: Iterable[str]) -> TopicPipeline:
        """Fits vectorizer and topic model with the given stream of texts.

        Parameters
        ----------
        texts: iterable of str
            Stream of texts to fit the pipeline with.

        Returns
        -------
        TopicPipeline
            Fitted pipeline.
        """
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.topic_model.fit(doc_term_matrix)
        return self

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        """Fits vectorizer and topic model and transforms the given text.

        Parameters
        ----------
        texts: iterable of str
            Texts to transform.

        Returns
        -------
        array of shape (n_documents, n_components)
            Document-topic matrix.
        """
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        return self.topic_model.fit_transform(doc_term_matrix)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Transforms given texts with the fitted pipeline.

        Parameters
        ----------
        texts: iterable of str
            Texts to transform.

        Returns
        -------
        array of shape (n_documents, n_components)
            Document-topic matrix.
        """
        doc_term_matrix = self.vectorizer.transform(texts)
        return self.topic_model.transform(doc_term_matrix)

    def visualize(self, texts: Iterable[str]):
        """Visualizes the model with pyLDAvis for inspection of the different
        mixture components :)

        Parameters
        ----------
        texts: iterable of str
            Texts to visualize the model with.

        Raises
        ------
        ModuleNotFoundError
            If pyLDAvis is not installed an exception is thrown.
        """
        try:
            import pyLDAvis
            import pyLDAvis.sklearn
        except ModuleNotFoundError as exception:
            raise ImportError(
                "Optional dependency pyLDAvis not installed.",
            ) from exception
        pyLDAvis.enable_notebook()
        return pyLDAvis.sklearn.prepare(
            self.topic_model,
            self.vectorizer.transform(texts),
            self.vectorizer,
        )

    def visualise(self, texts: Iterable[str]):
        """Alias of :meth:`~tweetopic.pipeline.TopicPipeline.visualize` for
        those living on this side of the Pacific."""
        return self.visualize

    def top_words(self, top_n: int | None = 10) -> list[dict[str, int]]:
        """Calculates the top words for each cluster.

        Parameters
        ----------
        top_n: int or None, default 10
            Top N words to return. If None, all words are returned.

        Returns
        -------
        list of dict of str to int
            Dictionary for each cluster mapping the words to number of occurances.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        dist: np.ndarray = self.topic_model.components_  # type: ignore
        res = []
        for i_cluster in range(self.topic_model.n_components):  # type: ignore
            top_indices = np.argsort(-dist[i_cluster])
            if top_n is not None:
                top_indices = top_indices[:top_n]  # type: ignore
            top_words = {
                feature_names[i]: dist[i_cluster, i]
                for i in top_indices
                if dist[
                    i_cluster,
                    i,
                ]  # Only return words if they are actually in the cluster
            }
            res.append(top_words)
        return res
