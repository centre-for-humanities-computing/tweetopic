from typing import Tuple

import numpy as np
import scipy.sparse as spr
from numba import njit


def init_doc_words(
    doc_term_matrix: spr.lil_matrix,
    max_unique_words: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_docs, _ = doc_term_matrix.shape
    doc_unique_words = np.zeros((n_docs, max_unique_words)).astype(np.uint32)
    doc_unique_word_counts = np.zeros((n_docs, max_unique_words)).astype(np.uint32)
    for i_doc in range(n_docs):
        unique_words = doc_term_matrix[i_doc].rows[0]  # type: ignore
        unique_word_counts = doc_term_matrix[i_doc].data[0]  # type: ignore
        for i_unique in range(len(unique_words)):
            doc_unique_words[i_doc, i_unique] = unique_words[i_unique]
            doc_unique_word_counts[i_doc, i_unique] = unique_word_counts[i_unique]
    return doc_unique_words, doc_unique_word_counts


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
