from typing import Tuple

import numpy as np
import scipy.sparse as spr


def init_doc_words(
    doc_term_matrix: spr.lil_matrix,
    max_unique_words: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_docs, _ = doc_term_matrix.shape
    doc_unique_words = np.zeros((n_docs, max_unique_words)).astype(np.uint32)
    doc_unique_word_counts = np.zeros((n_docs, max_unique_words)).astype(
        np.uint32
    )
    for i_doc in range(n_docs):
        unique_words = doc_term_matrix[i_doc].rows[0]  # type: ignore
        unique_word_counts = doc_term_matrix[i_doc].data[0]  # type: ignore
        for i_unique in range(len(unique_words)):
            doc_unique_words[i_doc, i_unique] = unique_words[i_unique]
            doc_unique_word_counts[i_doc, i_unique] = unique_word_counts[
                i_unique
            ]
    return doc_unique_words, doc_unique_word_counts
