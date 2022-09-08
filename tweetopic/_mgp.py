"""Module containing tools for fitting a Dirichlet Mixture Model."""
from __future__ import annotations

import numpy as np
from numba import njit

from tweetopic._doc import add_doc, remove_doc
from tweetopic._prob import predict_doc, sample_categorical


@njit
def init_clusters(
    cluster_word_distribution: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_doc_count: np.ndarray,
    doc_clusters: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    max_unique_words: int,
) -> None:
    """Randomly initializes clusters in the model.

    Parameters
    ----------
    cluster_word_count(OUT): array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution(OUT): matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    cluster_doc_count(OUT): array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    doc_clusters: array of shape (n_docs)
        Contains a cluster label for each document, that has
        to be assigned.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    max_unique_words: int
        Maximum number of unique words in any document.

    NOTE
    ----
    Beware that the function modifies a numpy array, that's passed in as
    an input parameter. Should not be used in parallel, as race conditions
    might arise.
    """
    n_docs, _ = doc_unique_words.shape
    for i_doc in range(n_docs):
        i_cluster = doc_clusters[i_doc]
        add_doc(
            i_doc=i_doc,
            i_cluster=i_cluster,
            cluster_word_distribution=cluster_word_distribution,
            cluster_word_count=cluster_word_count,
            cluster_doc_count=cluster_doc_count,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            max_unique_words=max_unique_words,
        )


@njit(parallel=False)
def fit_model(
    n_iter: int,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    doc_clusters: np.ndarray,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
) -> None:
    """Fits the Dirichlet Mixture Model with Gibbs Sampling. Implements
    algorithm described in Yin & Wang (2014)

    Parameters
    ----------
    n_iter: int
        Number of iterations to conduct.
    alpha: float
        Alpha parameter of the model.
    beta: float
        Beta parameter of the model.
    n_clusters: int
        Number of mixture components in the model.
    n_vocab: int
        Number of total vocabulary items.
    n_docs: int
        Total number of documents.
    doc_term_matrix: matrix of shape (n_documents, n_vocab)
        Contains how many times a term occurs in each document.
        (Bag of words matrix)
    doc_unique_words: array of shape (MAX_UNIQUE_WORDS, )
        Array containing all the ids of unique words in a document.
    doc_clusters: array of shape (n_docs, )
        Contains a cluster label for each document, that has
        to be assigned.
    doc_unique_words_count: array of shape (n_documents, )
        Vector containing the number of unique terms in each document.
    cluster_doc_count(OUT): array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    cluster_word_count(OUT): array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution(OUT): matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    max_unique_words: int
        Maximum count of unique words in a document seen in the corpus.
    """
    doc_word_count = np.sum(doc_unique_word_counts, axis=1)
    prediction = np.empty(n_clusters)
    for iteration in range(n_iter):
        total_transfers = 0
        for i_doc in range(n_docs):
            # Removing document from previous cluster
            prev_cluster = doc_clusters[i_doc]
            # Removing document from the previous cluster
            remove_doc(
                i_doc,
                prev_cluster,
                cluster_word_distribution=cluster_word_distribution,
                cluster_word_count=cluster_word_count,
                cluster_doc_count=cluster_doc_count,
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
                max_unique_words=max_unique_words,
            )
            # Getting new prediction for the document at hand
            predict_doc(
                probabilities=prediction,
                i_document=i_doc,
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
                n_words=doc_word_count[i_doc],
                alpha=alpha,
                beta=beta,
                n_clusters=n_clusters,
                n_vocab=n_vocab,
                n_docs=n_docs,
                cluster_doc_count=cluster_doc_count,
                cluster_word_count=cluster_word_count,
                cluster_word_distribution=cluster_word_distribution,
                max_unique_words=max_unique_words,
            )
            new_cluster = sample_categorical(prediction)
            if prev_cluster != new_cluster:
                total_transfers += 1
            # Adding document back to the newly chosen cluster
            doc_clusters[i_doc] = new_cluster
            add_doc(
                i_doc,
                new_cluster,
                cluster_word_distribution=cluster_word_distribution,
                cluster_word_count=cluster_word_count,
                cluster_doc_count=cluster_doc_count,
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
                max_unique_words=max_unique_words,
            )
        n_populated = np.count_nonzero(cluster_doc_count)
        print(
            f" Iteration {iteration}/{n_iter}: transferred {total_transfers} documents, {n_populated} clusters remain populated.",
        )
