"""Module containing tools for fitting a Dirichlet Multinomial Mixture Model."""
from __future__ import annotations

from math import exp, log

import numpy as np
from numba import njit
from tqdm import tqdm

from tweetopic._prob import norm_prob, sample_categorical


@njit
def _remove_add_doc(
    i_doc: int,
    i_cluster: int,
    remove: bool,
    cluster_word_distribution: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_doc_count: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    max_unique_words: int,
) -> None:
    if remove:
        cluster_doc_count[i_cluster] -= 1
    else:
        cluster_doc_count[i_cluster] += 1
    for i_unique in range(max_unique_words):
        i_word = doc_unique_words[i_doc, i_unique]
        count = doc_unique_word_counts[i_doc, i_unique]
        if not count:
            # Break out when the word is not present in the document
            break
        if remove:
            cluster_word_count[i_cluster] -= count
            cluster_word_distribution[i_cluster, i_word] -= count
        else:
            cluster_word_count[i_cluster] += count
            cluster_word_distribution[i_cluster, i_word] += count


@njit
def remove_doc(
    i_doc: int,
    i_cluster: int,
    cluster_word_distribution: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_doc_count: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    max_unique_words: int,
) -> None:
    return _remove_add_doc(
        i_doc,
        i_cluster,
        remove=True,
        cluster_word_distribution=cluster_word_distribution,
        cluster_word_count=cluster_word_count,
        cluster_doc_count=cluster_doc_count,
        doc_unique_words=doc_unique_words,
        doc_unique_word_counts=doc_unique_word_counts,
        max_unique_words=max_unique_words,
    )


@njit
def add_doc(
    i_doc: int,
    i_cluster: int,
    cluster_word_distribution: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_doc_count: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    max_unique_words: int,
) -> None:
    return _remove_add_doc(
        i_doc,
        i_cluster,
        remove=False,
        cluster_word_distribution=cluster_word_distribution,
        cluster_word_count=cluster_word_count,
        cluster_doc_count=cluster_doc_count,
        doc_unique_words=doc_unique_words,
        doc_unique_word_counts=doc_unique_word_counts,
        max_unique_words=max_unique_words,
    )


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


@njit(fastmath=True)
def _cond_prob(
    i_cluster: int,
    i_document: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    n_words: int,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
) -> float:
    """Computes the conditional probability of a certain document joining the
    given mixture component.

    Implements formula no. 4 from Yin & Wang (2014).

    Parameters
    ----------
    i_cluster: int
        The label of the cluster.
    i_document: int
        Index of the document in the corpus.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    n_words: int
        Total number of words in the document.
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
    cluster_doc_count: array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    cluster_word_count: array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution: matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    max_unique_words: int
        Maximum number of unique words seen in a document.
    """
    # I broke the formula into different pieces so that it's easier to write
    # I could not find a better way to organize it, as I'm not in total command of
    # the algebra going on here :))
    # I use logs instead of computing the products directly,
    # as it would quickly result in numerical overflow.
    log_norm_term = log(
        (cluster_doc_count[i_cluster] + alpha)
        / (n_docs - 1 + n_clusters * alpha),
    )
    log_numerator = 0
    for i_unique in range(max_unique_words):
        i_word = doc_unique_words[i_document, i_unique]
        count = doc_unique_word_counts[i_document, i_unique]
        if not count:
            # Breaking out at the first word that doesn't occur in the document
            break
        for j in range(count):
            log_numerator += log(
                cluster_word_distribution[i_cluster, i_word] + beta + j,
            )
    log_denominator = 0
    subres = cluster_word_count[i_cluster] + (n_vocab * beta)
    for j in range(n_words):
        log_denominator += log(subres + j)
    res = exp(log_norm_term + log_numerator - log_denominator)
    return res


@njit(parallel=False, fastmath=True)
def predict_doc(
    probabilities: np.ndarray,
    i_document: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    n_words: int,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
) -> None:
    """Computes the parameters of the multinomial distribution used for
    sampling.

    Parameters
    ----------
    probabilities(OUT): array of shape (n_clusters, )
        Parameters of the categorical distribution.
    i_document: int
        Index of the document in the corpus.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    n_words: int
        Total number of words in the document.
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
    cluster_doc_count: array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    cluster_word_count: array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution: matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    max_unique_words: int
        Maximum number of unique words seen in a document.

    NOTE
    ----
    Beware that the function modifies a numpy array, that's passed in as
    an input parameter. Should not be used in parallel, as race conditions
    might arise.
    """
    # NOTE: we modify the original array here instead of returning a new
    # one, as allocating new arrays in such a nested loop is very inefficient.
    # Obtain all conditional probabilities
    for i_cluster in range(n_clusters):
        probabilities[i_cluster] = _cond_prob(
            i_cluster=i_cluster,
            i_document=i_document,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            n_words=n_words,
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
    # Normalize probability vector
    norm_prob(probabilities)


@njit(parallel=False)
def _sampling_step(
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
    prediction: np.ndarray,
    doc_word_count: np.ndarray,
) -> None:
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
    doc_word_count = np.sum(doc_unique_word_counts, axis=1)
    prediction = np.empty(n_clusters)
    iterator = tqdm(range(n_iter), desc="Sampling")
    for _ in iterator:
        _sampling_step(
            alpha=alpha,
            beta=beta,
            n_clusters=n_clusters,
            n_vocab=n_vocab,
            n_docs=n_docs,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            doc_clusters=doc_clusters,
            cluster_doc_count=cluster_doc_count,
            cluster_word_count=cluster_word_count,
            cluster_word_distribution=cluster_word_distribution,
            max_unique_words=max_unique_words,
            doc_word_count=doc_word_count,
            prediction=prediction,
        )


#
# @njit(parallel=False)
# def fit_model(
#     n_iter: int,
#     alpha: float,
#     beta: float,
#     n_clusters: int,
#     n_vocab: int,
#     n_docs: int,
#     doc_unique_words: np.ndarray,
#     doc_unique_word_counts: np.ndarray,
#     doc_clusters: np.ndarray,
#     cluster_doc_count: np.ndarray,
#     cluster_word_count: np.ndarray,
#     cluster_word_distribution: np.ndarray,
#     max_unique_words: int,
# ) -> None:
#     """Fits the Dirichlet Mixture Model with Gibbs Sampling. Implements
#     algorithm described in Yin & Wang (2014)
#
#     Parameters
#     ----------
#     n_iter: int
#         Number of iterations to conduct.
#     alpha: float
#         Alpha parameter of the model.
#     beta: float
#         Beta parameter of the model.
#     n_clusters: int
#         Number of mixture components in the model.
#     n_vocab: int
#         Number of total vocabulary items.
#     n_docs: int
#         Total number of documents.
#     doc_term_matrix: matrix of shape (n_documents, n_vocab)
#         Contains how many times a term occurs in each document.
#         (Bag of words matrix)
#     doc_unique_words: array of shape (MAX_UNIQUE_WORDS, )
#         Array containing all the ids of unique words in a document.
#     doc_clusters: array of shape (n_docs, )
#         Contains a cluster label for each document, that has
#         to be assigned.
#     doc_unique_words_count: array of shape (n_documents, )
#         Vector containing the number of unique terms in each document.
#     cluster_doc_count(OUT): array of shape (n_clusters,)
#         Array containing how many documents there are in each cluster.
#     cluster_word_count(OUT): array of shape (n_clusters,)
#         Contains the amount of words there are in each cluster.
#     cluster_word_distribution(OUT): matrix of shape (n_clusters, n_vocab)
#         Contains the amount a word occurs in a certain cluster.
#     max_unique_words: int
#         Maximum count of unique words in a document seen in the corpus.
#     """
#     doc_word_count = np.sum(doc_unique_word_counts, axis=1)
#     prediction = np.empty(n_clusters)
#     for iteration in range(n_iter):
#         total_transfers = 0
#         for i_doc in range(n_docs):
#             # Removing document from previous cluster
#             prev_cluster = doc_clusters[i_doc]
#             # Removing document from the previous cluster
#             remove_doc(
#                 i_doc,
#                 prev_cluster,
#                 cluster_word_distribution=cluster_word_distribution,
#                 cluster_word_count=cluster_word_count,
#                 cluster_doc_count=cluster_doc_count,
#                 doc_unique_words=doc_unique_words,
#                 doc_unique_word_counts=doc_unique_word_counts,
#                 max_unique_words=max_unique_words,
#             )
#             # Getting new prediction for the document at hand
#             predict_doc(
#                 probabilities=prediction,
#                 i_document=i_doc,
#                 doc_unique_words=doc_unique_words,
#                 doc_unique_word_counts=doc_unique_word_counts,
#                 n_words=doc_word_count[i_doc],
#                 alpha=alpha,
#                 beta=beta,
#                 n_clusters=n_clusters,
#                 n_vocab=n_vocab,
#                 n_docs=n_docs,
#                 cluster_doc_count=cluster_doc_count,
#                 cluster_word_count=cluster_word_count,
#                 cluster_word_distribution=cluster_word_distribution,
#                 max_unique_words=max_unique_words,
#             )
#             new_cluster = sample_categorical(prediction)
#             if prev_cluster != new_cluster:
#                 total_transfers += 1
#             # Adding document back to the newly chosen cluster
#             doc_clusters[i_doc] = new_cluster
#             add_doc(
#                 i_doc,
#                 new_cluster,
#                 cluster_word_distribution=cluster_word_distribution,
#                 cluster_word_count=cluster_word_count,
#                 cluster_doc_count=cluster_doc_count,
#                 doc_unique_words=doc_unique_words,
#                 doc_unique_word_counts=doc_unique_word_counts,
#                 max_unique_words=max_unique_words,
#             )
#         n_populated = np.count_nonzero(cluster_doc_count)
#         print(
#             f" Iteration {iteration}/{n_iter}: transferred"
#             f" {total_transfers} documents,"
#             f"{n_populated} clusters remain populated.",
#         )
