from __future__ import annotations

from typing import Iterable, Optional
from math import log, exp
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from numba import njit, prange

from tweetopic.exceptions import NotFittedException


@njit(fastmath=True)
def _rand_uniform(size: int) -> np.ndarray:
    res = np.empty(size)
    for i in range(size):
        res[i] = random.random()  # np.random.uniform(0.0, 1.0)
    return res


@njit(fastmath=True)
def _sample_categorical(pvals: np.ndarray) -> int:
    cum_prob = np.cumsum(pvals)
    return np.argmax(_rand_uniform(pvals.shape[0]) < cum_prob)  # type: ignore


# @njit(fastmath=True)
# def _sample_categorical(pvals: np.ndarray) -> int:
#     # Rejection sampling with cummulutative probabilities :)
#     cum_prob = 0
#     for i in prange(len(pvals)):
#         cum_prob += pvals[i]
#         if np.random.uniform(0.0, 1.0) < cum_prob:
#             return i
#     else:
#         return 0


@njit
def _init_clusters(
    cluster_word_count: np.ndarray,
    doc_clusters: np.ndarray,
    doc_term_matrix: np.ndarray,
) -> None:
    print("Initializing clusters:")
    for cluster, document in zip(doc_clusters, doc_term_matrix):
        cluster_word_count[cluster] += document


@njit(fastmath=True)
def _cond_prob(
    i_cluster: int,
    document: np.ndarray,
    alpha: float,
    beta: float,
    n_words: int,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
) -> float:
    # I broke the formula into different pieces so that it's easier to write
    # I could not find a better way to organize it, as I'm not in total command of
    # the algebra going on here :))
    log_norm_term = log(
        (cluster_doc_count[i_cluster] + alpha) / (n_docs - 1 + n_clusters * alpha)
    )
    log_numerator = 0
    for i_word in range(n_vocab):
        n_i_word = document[i_word]
        subres = cluster_word_distribution[i_cluster, i_word] + beta
        for j in range(n_i_word):
            log_numerator += log(subres + j)
    log_denominator = 0
    # n_words = sum(document)
    subres = cluster_word_count[i_cluster] + (n_vocab * beta)
    for j in range(n_words):
        log_denominator += log(subres + j)
    res = exp(log_norm_term + log_numerator - log_denominator)
    return res


@njit(parallel=True, fastmath=True)
def _predict_doc(
    document: np.ndarray,
    alpha: float,
    beta: float,
    n_words: int,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
) -> np.ndarray:
    # Obtain all conditional probabilities
    probabilities = np.zeros(n_clusters)
    for i_cluster in prange(n_clusters):
        probabilities[i_cluster] = _cond_prob(
            i_cluster,
            document,
            alpha,
            beta,
            n_words,
            n_clusters,
            n_vocab,
            n_docs,
            cluster_doc_count,
            cluster_word_count,
            cluster_word_distribution,
        )
    # Normalize probability vector
    norm_term = sum(probabilities) or 1
    probabilities = probabilities / norm_term
    return probabilities


@njit(fastmath=True)
def _fit_model(
    n_iter: int,
    doc_term_matrix: np.ndarray,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    doc_clusters: np.ndarray,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
) -> None:
    doc_word_count = np.sum(doc_term_matrix, axis=1)
    for iteration in range(n_iter):
        print(f"Iteration no. {iteration}")
        total_transfers = 0
        for i_doc in range(doc_term_matrix.shape[0]):
            document = doc_term_matrix[i_doc]
            # Removing document from previous cluster
            prev_cluster = doc_clusters[i_doc]
            cluster_doc_count[prev_cluster] -= 1
            cluster_word_count[prev_cluster] -= doc_word_count[i_doc]
            cluster_word_distribution[prev_cluster, :] -= doc_term_matrix[i_doc, :]
            # Getting new prediction for the document at hand
            prediction = _predict_doc(
                document,
                alpha,
                beta,
                doc_word_count[i_doc],
                n_clusters,
                n_vocab,
                n_docs,
                cluster_doc_count,
                cluster_word_count,
                cluster_word_distribution,
            )
            new_cluster = _sample_categorical(prediction)
            if prev_cluster != new_cluster:
                total_transfers += 1
            # Adding document back to the newly chosen cluster
            doc_clusters[i_doc] = new_cluster
            cluster_doc_count[new_cluster] += 1
            cluster_word_count[new_cluster] += doc_word_count[i_doc]
            cluster_word_distribution[new_cluster, :] += doc_term_matrix[i_doc, :]
        n_populated = np.count_nonzero(cluster_doc_count)
        print(f"    {n_populated} out of {n_clusters} clusters remain populated.")
        if not total_transfers:
            print("    The model converged, stopping iterations.")
            break
        else:
            print(f"    Transferred {total_transfers} in iteration no. {iteration}.")


class MovieGroupProcess:
    def __init__(self, n_clusters: int = 8, alpha: float = 0.1, beta: float = 0.1):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.cluster_word_distribution: Optional[np.ndarray] = None
        self.cluster_doc_count: Optional[np.ndarray] = None
        self.cluster_word_count: Optional[np.ndarray] = None
        self.vectorizer: Optional[CountVectorizer] = None
        self.n_documents = 0
        self.n_vocab = 0

    def fit(
        self,
        documents: Iterable[str],
        n_iterations: int = 30,
        **vectorizer_kwargs,
    ) -> MovieGroupProcess:
        print("Converting documents to BOW matrix.")
        self.vectorizer = CountVectorizer(**vectorizer_kwargs)
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        # TODO: figure out a better way to do this, this is mildly stupid
        doc_term_matrix = doc_term_matrix.toarray()
        self.n_documents, self.n_vocab = doc_term_matrix.shape
        initial_clusters = np.random.multinomial(
            1, np.ones(self.n_clusters) / self.n_clusters, size=self.n_documents
        )
        doc_clusters = np.argmax(initial_clusters, axis=1)
        self.cluster_doc_count = initial_clusters.sum(axis=0)
        self.cluster_word_distribution = np.zeros((self.n_clusters, self.n_vocab))
        print("Initialising mixture components")
        _init_clusters(self.cluster_word_distribution, doc_clusters, doc_term_matrix)
        self.cluster_word_count = np.sum(self.cluster_word_distribution, axis=1)
        print("Fitting model")
        _fit_model(
            n_iter=n_iterations,
            doc_term_matrix=doc_term_matrix,
            alpha=self.alpha,
            beta=self.beta,
            n_clusters=self.n_clusters,
            n_vocab=self.n_vocab,
            n_docs=self.n_documents,
            doc_clusters=doc_clusters,
            cluster_doc_count=self.cluster_doc_count,
            cluster_word_count=self.cluster_word_count,
            cluster_word_distribution=self.cluster_word_distribution,
        )
        return self

    def predict(self, documents: Iterable[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise NotFittedException("MovieGroupProcess: Model was not fitted.")
        embeddings = self.vectorizer.transform(documents).toarray()
        predictions = []
        for doc in embeddings:
            pred = _predict_doc(
                document=doc,
                alpha=self.alpha,
                beta=self.beta,
                n_clusters=self.n_clusters,
                n_vocab=self.n_vocab,
                n_docs=self.n_documents,
                cluster_doc_count=self.cluster_doc_count,  # type: ignore
                cluster_word_count=self.cluster_word_count,  # type: ignore
                cluster_word_distribution=self.cluster_word_distribution,  # type: ignore
            )
            predictions.append(pred)
        return np.stack(predictions)
