"""Module for utility functions for fitting BTMs"""

import random
from typing import Dict, Tuple, TypeVar

import numpy as np
from numba import njit
from tqdm import tqdm

from tweetopic._prob import norm_prob, sample_categorical


@njit
def doc_unique_biterms(
    doc_unique_words: np.ndarray, doc_unique_word_counts: np.ndarray
) -> Dict[Tuple[int, int], int]:
    (n_max_unique_words,) = doc_unique_words.shape
    biterm_counts = dict()
    for i_term in range(n_max_unique_words):
        count_i = doc_unique_word_counts[i_term]
        term_i = doc_unique_words[i_term]
        if not count_i:
            break
        for j_term in range(n_max_unique_words):
            count_j = doc_unique_word_counts[j_term]
            term_j = doc_unique_words[j_term]
            if not count_j:
                break
            biterm = min(term_i, term_j), max(term_i, term_j)
            if biterm in biterm_counts:
                continue
            if term_j == term_i:
                count = count_i - 1
                if count:
                    biterm_counts[biterm] = count
            else:
                biterm_counts[biterm] = count_i * count_j
    return biterm_counts


T = TypeVar("T")


@njit
def nb_add_counter(dest: Dict[T, int], source: Dict[T, int]):
    """Adds one counter dict to another in place with Numba"""
    for key in source:
        if key in dest:
            dest[key] += source[key]
        else:
            dest[key] = source[key]


@njit
def corpus_unique_biterms(
    doc_unique_words: np.ndarray, doc_unique_word_counts: np.ndarray
) -> Dict[Tuple[int, int], int]:
    n_documents, _ = doc_unique_words.shape
    biterm_counts = doc_unique_biterms(
        doc_unique_words[0], doc_unique_word_counts[0]
    )
    for i_doc in range(1, n_documents):
        doc_unique_words_i = doc_unique_words[i_doc]
        doc_unique_word_counts_i = doc_unique_word_counts[i_doc]
        doc_biterms = doc_unique_biterms(
            doc_unique_words_i, doc_unique_word_counts_i
        )
        nb_add_counter(biterm_counts, doc_biterms)
    return biterm_counts


@njit
def compute_biterm_set(
    biterm_counts: Dict[Tuple[int, int], int]
) -> np.ndarray:
    return np.array(list(biterm_counts.keys()))


@njit
def arrayify_biterms(
    unique_biterms: np.ndarray,
    unique_biterm_counts: np.ndarray,
    biterm_counts: Dict[Tuple[int, int], int],
):
    i_biterm = 0
    for biterm, count in biterm_counts.items():
        w_i, w_j = biterm
        unique_biterms[i_biterm, 0] = w_i
        unique_biterms[i_biterm, 1] = w_j
        unique_biterm_counts[i_biterm] = count
        i_biterm += 1
    unique_biterm_counts[i_biterm] = 0


@njit(fastmath=True)
def add_remove_biterm(
    add: bool,
    i_biterm: int,
    i_topic: int,
    biterms: np.ndarray,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
) -> None:
    change = 1 if add else -1
    topic_word_count[i_topic, biterms[i_biterm, 0]] += change
    topic_word_count[i_topic, biterms[i_biterm, 1]] += change
    topic_biterm_count[i_topic] += change


@njit(fastmath=True)
def add_biterm(
    i_biterm: int,
    i_topic: int,
    biterms: np.ndarray,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
) -> None:
    add_remove_biterm(
        True, i_biterm, i_topic, biterms, topic_word_count, topic_biterm_count
    )


@njit(fastmath=True)
def remove_biterm(
    i_biterm: int,
    i_topic: int,
    biterms: np.ndarray,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
) -> None:
    add_remove_biterm(
        False, i_biterm, i_topic, biterms, topic_word_count, topic_biterm_count
    )


@njit(fastmath=True)
def init_components(
    n_components: int,
    n_vocab: int,
    biterms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_biterms, _ = biterms.shape
    topic_word_count = np.zeros((n_components, n_vocab), dtype=np.uint)
    topic_biterm_count = np.zeros(n_components, dtype=np.uint)
    biterm_topic_assignments = np.empty(n_biterms, dtype=np.uint)
    for i_biterm in range(n_biterms):
        i_topic = random.randint(0, n_components - 1)
        biterm_topic_assignments[i_biterm] = i_topic
        add_biterm(
            i_biterm, i_topic, biterms, topic_word_count, topic_biterm_count
        )
    return biterm_topic_assignments, topic_word_count, topic_biterm_count


@njit(fastmath=True)
def _topic_biterm_probability(
    i_topic: int,
    i_biterm: int,
    alpha: float,
    beta: float,
    n_vocab: int,
    biterms: np.ndarray,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
) -> float:
    term_i = biterms[i_biterm, 0]
    term_j = biterms[i_biterm, 1]
    numerator = (
        (topic_biterm_count[i_topic] + alpha)
        * (topic_word_count[i_topic, term_i] + beta)
        * (topic_word_count[i_topic, term_j] + beta)
    )
    n_topic_words = topic_biterm_count[i_topic] * 2
    denominator = (n_topic_words + n_vocab * beta) ** 2
    return numerator / denominator


@njit(fastmath=True)
def propose_topic_biterm(
    prediction: np.ndarray,
    i_biterm: int,
    n_components: int,
    alpha: float,
    beta: float,
    n_vocab: int,
    biterms: np.ndarray,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
) -> None:
    for i_topic in range(n_components):
        prediction[i_topic] = _topic_biterm_probability(
            i_topic=i_topic,
            i_biterm=i_biterm,
            alpha=alpha,
            beta=beta,
            n_vocab=n_vocab,
            biterms=biterms,
            topic_word_count=topic_word_count,
            topic_biterm_count=topic_biterm_count,
        )
    norm_prob(prediction)


@njit(fastmath=True)
def estimate_parameters(
    alpha: float,
    beta: float,
    n_components: int,
    n_vocab: int,
    n_biterms: int,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
):
    topic_word_distribution = np.empty((n_components, n_vocab))
    topic_distribution = np.empty(n_components)
    # Equation 6
    for i_topic in range(n_components):
        topic_distribution[i_topic] = (topic_biterm_count[i_topic] + alpha) / (
            n_biterms + n_components * alpha
        )
    # Equation 5
    for i_topic in range(n_components):
        n_topic_terms = topic_biterm_count[i_topic] * 2
        for i_term in range(n_vocab):
            topic_word_distribution[i_topic, i_term] = (
                topic_word_count[i_topic, i_term] + beta
            ) / (n_topic_terms + n_vocab * beta)
    return topic_distribution, topic_word_distribution


@njit(fastmath=True)
def _sampling_step(
    alpha: float,
    beta: float,
    n_components: int,
    n_vocab: int,
    biterms: np.ndarray,
    n_biterms: int,
    biterm_topic_assignments: np.ndarray,
    topic_word_count: np.ndarray,
    topic_biterm_count: np.ndarray,
    prediction: np.ndarray,
):
    for i_biterm in range(n_biterms):
        prev_topic = biterm_topic_assignments[i_biterm]
        remove_biterm(
            i_biterm=i_biterm,
            i_topic=prev_topic,
            biterms=biterms,
            topic_word_count=topic_word_count,
            topic_biterm_count=topic_biterm_count,
        )
        propose_topic_biterm(
            prediction=prediction,
            i_biterm=i_biterm,
            n_components=n_components,
            alpha=alpha,
            beta=beta,
            n_vocab=n_vocab,
            biterms=biterms,
            topic_word_count=topic_word_count,
            topic_biterm_count=topic_biterm_count,
        )
        next_topic = sample_categorical(prediction)
        add_biterm(
            i_biterm=i_biterm,
            i_topic=next_topic,
            biterms=biterms,
            topic_word_count=topic_word_count,
            topic_biterm_count=topic_biterm_count,
        )
        biterm_topic_assignments[i_biterm] = next_topic


def fit_model(
    n_iter: int,
    alpha: float,
    beta: float,
    n_components: int,
    n_vocab: int,
    biterms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    (
        biterm_topic_assignments,
        topic_word_count,
        topic_biterm_count,
    ) = init_components(n_components, n_vocab, biterms)
    n_biterms, _ = biterms.shape
    prediction = np.zeros(n_components)
    iterator = tqdm(range(n_iter), desc="Sampling")
    for _ in iterator:
        _sampling_step(
            alpha=alpha,
            beta=beta,
            n_components=n_components,
            n_vocab=n_vocab,
            biterms=biterms,
            n_biterms=n_biterms,
            biterm_topic_assignments=biterm_topic_assignments,
            topic_word_count=topic_word_count,
            topic_biterm_count=topic_biterm_count,
            prediction=prediction,
        )
    topic_distribution, topic_word_distribution = estimate_parameters(
        alpha=alpha,
        beta=beta,
        n_components=n_components,
        n_vocab=n_vocab,
        n_biterms=n_biterms,
        topic_word_count=topic_word_count,
        topic_biterm_count=topic_biterm_count,
    )
    return topic_distribution, topic_word_distribution


#
# @njit(fastmath=True)
# def fit_model(
#     n_iter: int,
#     alpha: float,
#     beta: float,
#     n_components: int,
#     n_vocab: int,
#     biterms: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     (
#         biterm_topic_assignments,
#         topic_word_count,
#         topic_biterm_count,
#     ) = init_components(n_components, n_vocab, biterms)
#     n_biterms, _ = biterms.shape
#     prediction = np.zeros(n_components)
#     for i_iter in range(n_iter):
#         n_transferred = 0
#         for i_biterm in range(n_biterms):
#             prev_topic = biterm_topic_assignments[i_biterm]
#             remove_biterm(
#                 i_biterm=i_biterm,
#                 i_topic=prev_topic,
#                 biterms=biterms,
#                 topic_word_count=topic_word_count,
#                 topic_biterm_count=topic_biterm_count,
#             )
#             propose_topic_biterm(
#                 prediction=prediction,
#                 i_biterm=i_biterm,
#                 n_components=n_components,
#                 alpha=alpha,
#                 beta=beta,
#                 n_vocab=n_vocab,
#                 biterms=biterms,
#                 topic_word_count=topic_word_count,
#                 topic_biterm_count=topic_biterm_count,
#             )
#             next_topic = sample_categorical(prediction)
#             add_biterm(
#                 i_biterm=i_biterm,
#                 i_topic=next_topic,
#                 biterms=biterms,
#                 topic_word_count=topic_word_count,
#                 topic_biterm_count=topic_biterm_count,
#             )
#             biterm_topic_assignments[i_biterm] = next_topic
#             # NOTE: Consider branchless, if it affects performance
#             # n_tranferred += int(next_topic != prev_topic)
#             if next_topic != prev_topic:
#                 n_transferred += 1
#         print(f" Iteration {i_iter}: transferred {n_transferred} biterms.")
#     topic_distribution, topic_word_distribution = estimate_parameters(
#         alpha=alpha,
#         beta=beta,
#         n_components=n_components,
#         n_vocab=n_vocab,
#         n_biterms=n_biterms,
#         topic_word_count=topic_word_count,
#         topic_biterm_count=topic_biterm_count,
#     )
#     return topic_distribution, topic_word_distribution
#


@njit(fastmath=True)
def prob_topic_given_biterm(
    w_i: int,
    w_j: int,
    prediction: np.ndarray,
    topic_distribution: np.ndarray,
    topic_word_distribution: np.ndarray,
):
    """Predicts probabilities of topics given biterms."""
    n_topics = topic_distribution.shape[0]
    for i_topic in range(n_topics):
        prediction[i_topic] = (
            topic_distribution[i_topic]
            * topic_word_distribution[i_topic, w_i]
            * topic_word_distribution[i_topic, w_j]
        )
    norm_prob(prediction)


@njit(fastmath=True)
def prob_topic_given_document(
    prediction: np.ndarray,
    doc_biterms: Dict[Tuple[int, int], int],
    topic_distribution: np.ndarray,
    topic_word_distribution: np.ndarray,
):
    """Predicts probabilities of all topics given biterms in a document."""
    n_topics = prediction.shape[0]
    total_biterms = 0
    for biterm_count in doc_biterms.values():
        total_biterms += biterm_count
    p_topic_given_biterm = np.empty(n_topics)
    for i_topic in range(n_topics):
        prediction[i_topic] = 0
        for biterm, biterm_count in doc_biterms.items():
            w_i, w_j = biterm
            prob_topic_given_biterm(
                w_i,
                w_j,
                p_topic_given_biterm,
                topic_distribution,
                topic_word_distribution,
            )
            p_biterm_given_document = biterm_count / total_biterms
            prediction[i_topic] += (
                p_topic_given_biterm[i_topic] * p_biterm_given_document
            )
    norm_prob(prediction)


# TODO: Something wrong with this
@njit(fastmath=True)
def predict_docs(
    topic_distribution: np.ndarray,
    topic_word_distribution: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
) -> np.ndarray:
    n_docs = doc_unique_words.shape[0]
    n_topics = topic_distribution.shape[0]
    predictions = np.empty((n_docs, n_topics))
    pred = np.empty(n_topics)
    for i_doc in range(n_docs):
        words, word_counts = (
            doc_unique_words[i_doc],
            doc_unique_word_counts[i_doc],
        )
        biterms = doc_unique_biterms(words, word_counts)
        prob_topic_given_document(
            pred, biterms, topic_distribution, topic_word_distribution
        )
        predictions[i_doc, :] = pred
    return predictions


# @njit(fastmath=True)
# def prob_topic_biterm(
#     i_topic: int,
#     biterm: np.ndarray,
#     n_components: int,
#     topic_distribution: np.ndarray,
#     topic_word_distribution: np.ndarray,
# ) -> float:
#     term_i, term_j = biterm
#     evidence = 0
#     for j_topic in range(n_components):
#         # Calculating this everytime is mildly stupid,
#         # should probably come up with sth smarter.
#         evidence += (
#             topic_distribution[j_topic]
#             * topic_word_distribution[j_topic, term_i]
#             * topic_word_distribution[j_topic, term_j]
#         )
#     prior = topic_distribution[i_topic]
#     posterior = (
#         topic_word_distribution[i_topic, term_i]
#         * topic_word_distribution[i_topic, term_j]
#     )
#     if evidence:
#         return prior * posterior / evidence
#     else:
#         return 0
#
#
# @njit(fastmath=True)
# def predict_doc(
#     n_components: int,
#     topic_distribution: np.ndarray,
#     topic_word_distribution: np.ndarray,
#     doc_unique_biterms: np.ndarray,
#     doc_unique_biterm_counts: np.ndarray,
# ) -> np.ndarray:
#     prediction = np.zeros(n_components)
#     n_biterms = np.sum(doc_unique_biterm_counts)
#     n_max_unique_biterms, _ = doc_unique_biterms.shape
#     for i_topic in range(n_components):
#         i_biterm = 0
#         while (
#             i_biterm < n_max_unique_biterms
#             and doc_unique_biterm_counts[i_biterm]
#         ):
#             biterm = doc_unique_biterms[i_biterm]
#             if n_biterms:
#                 prob_biterm_document = (
#                     doc_unique_biterm_counts[i_biterm] / n_biterms
#                 )
#             else:
#                 prob_biterm_document = 0
#             prediction += (
#                 prob_topic_biterm(
#                     i_topic,
#                     biterm,
#                     n_components,
#                     topic_distribution,
#                     topic_word_distribution,
#                 )
#                 * prob_biterm_document
#             )
#             i_biterm += 1
#     return prediction
#
