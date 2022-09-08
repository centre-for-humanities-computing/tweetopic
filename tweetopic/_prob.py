from math import exp, log

import numpy as np
from numba import njit


@njit(parallel=False)
def sample_categorical(pvals: np.ndarray) -> int:
    """Samples from a categorical distribution given its parameters.

    Parameters
    ----------
    pvals: array of shape (n_clusters, )
        Parameters of the categorical distribution.

    Returns
    -------
    int
        Sample.
    """
    # NOTE: This function was needed as numba's implementation
    # of numpy's multinomial sampling function has some floating point shenanigans going on.
    # Rejection sampling with cummulutative probabilities :)
    cum_prob = 0
    u = np.random.uniform(0.0, 1.0)
    for i in range(len(pvals)):
        cum_prob += pvals[i]
        if u < cum_prob:
            return i
    else:
        # This shouldn't ever happen, but floating point errors can
        # cause such behaviour ever so often.
        return 0


@njit(parallel=False)
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
        (cluster_doc_count[i_cluster] + alpha) / (n_docs - 1 + n_clusters * alpha),
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


@njit(parallel=False)
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
    norm_term = sum(probabilities) or 1
    probabilities[:] = probabilities / norm_term
