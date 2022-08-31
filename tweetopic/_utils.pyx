cimport numpy

def _rand_uniform(size: int) -> np.ndarray:
    res = np.empty(size)
    for i in range(size):
        res[i] = np.random.uniform(0.0, 1.0)
    return res



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
