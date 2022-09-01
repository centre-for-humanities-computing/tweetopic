from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from numba import njit
from sklearn.feature_extraction.text import CountVectorizer

from tweetopic.exceptions import NotFittedException
from tweetopic._prob import sample_categorical, predict_doc
from tweetopic._doc import MAX_UNIQUE_WORDS, init_doc_words, remove_doc, add_doc


@njit
def _init_clusters(
    cluster_word_distribution: np.ndarray,
    doc_clusters: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
) -> None:
    """
    Randomly initializes clusters in the model.

    Parameters
    ----------
    cluster_word_distribution(OUT): matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    doc_clusters: array of shape (n_docs)
        Contains a cluster label for each document, that has
        to be assigned.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    NOTE
    ----
    Beware that the function modifies a numpy array, that's passed in as
    an input parameter. Should not be used in parallel, as race conditions
    might arise.
    """
    n_docs, _ = doc_unique_words.shape
    for cluster, i_doc in zip(doc_clusters, range(n_docs)):
        for i_unique in range(MAX_UNIQUE_WORDS):
            i_word = doc_unique_words[i_doc, i_unique]
            count = doc_unique_word_counts[i_doc, i_unique]
            if not count:
                break
            cluster_word_distribution[cluster, i_word] += count


@njit(parallel=False)
def _fit_model(
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
) -> None:
    """
    Fits the Dirichlet Mixture Model with Gibbs Sampling.
    Implements algorithm described in Yin & Wang (2014)

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
    """
    doc_word_count = np.sum(doc_unique_word_counts, axis=1)
    prediction = np.empty(n_clusters)
    for iteration in range(n_iter):
        print(f"Iteration no. {iteration}")
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
            )
        n_populated = np.count_nonzero(cluster_doc_count)
        print(f"    {n_populated} out of {n_clusters} clusters remain populated.")
        if not total_transfers:
            print("    The model converged, stopping iterations.")
            break
        else:
            print(f"    Transferred {total_transfers} documents.")


class MovieGroupProcess:
    """
    Class for fitting a dirichlet mixture model with the movie group process algorithm
    described in Yin & Wang's paper (2014).

    Hyperparameters
    ---------------
    n_clusters: int, default 8
        Number of mixture components in the model.
    alpha: float, default 0.1
        Willingness of a document joining an empty cluster.
    beta: float, default 0.1
        Willingness to join clusters, where the terms in the document
        are not present.

    Attributes
    ----------
    cluster_word_distribution: matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    cluster_doc_count: array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    cluster_word_count: array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    vectorizer: CountVectorizer
        BOW vectorizer from sklearn.
        It is used to transform documents into bag of words embeddings.
    n_vocab: int
        Number of total vocabulary items seen during fitting.
    n_documents: int
        Total number of documents seen during fitting.
    """

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
        # doc_term_matrix = doc_term_matrix_sparse.toarray().astype(np.uint8)
        self.n_documents, self.n_vocab = doc_term_matrix.shape
        print("Calculating unique words.")
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            doc_term_matrix.tolil()
        )
        print("Initialising mixture components")
        initial_clusters = np.random.multinomial(
            1, np.ones(self.n_clusters) / self.n_clusters, size=self.n_documents
        )
        doc_clusters = np.argmax(initial_clusters, axis=1)
        self.cluster_doc_count = initial_clusters.sum(axis=0)
        self.cluster_word_distribution = np.zeros((self.n_clusters, self.n_vocab))
        _init_clusters(
            self.cluster_word_distribution,
            doc_clusters,
            doc_unique_words,
            doc_unique_word_counts,
        )
        self.cluster_word_count = np.sum(self.cluster_word_distribution, axis=1)
        print("Fitting model")
        _fit_model(
            n_iter=n_iterations,
            alpha=self.alpha,
            beta=self.beta,
            n_clusters=self.n_clusters,
            n_vocab=self.n_vocab,
            n_docs=self.n_documents,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            doc_clusters=doc_clusters,
            cluster_doc_count=self.cluster_doc_count,
            cluster_word_count=self.cluster_word_count,
            cluster_word_distribution=self.cluster_word_distribution,
        )
        return self

    def predict(self, documents: Iterable[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise NotFittedException("MovieGroupProcess: Model was not fitted.")
        embeddings = self.vectorizer.transform(documents)
        doc_unique_words, doc_unique_word_counts = init_doc_words(embeddings.tolil())
        doc_words_count = np.sum(doc_unique_word_counts, axis=1)
        n_docs = embeddings.shape[0]
        predictions = []
        for i_doc in range(n_docs):
            pred = np.zeros(self.n_clusters)
            predict_doc(
                probabilities=pred,
                i_document=i_doc,
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
                n_words=doc_words_count[i_doc],
                alpha=self.alpha,
                beta=self.beta,
                n_clusters=self.n_clusters,
                n_vocab=self.n_vocab,
                n_docs=n_docs,
                cluster_doc_count=self.cluster_doc_count,  # type: ignore
                cluster_word_count=self.cluster_word_count,  # type: ignore
                cluster_word_distribution=self.cluster_word_distribution,  # type: ignore
            )
            predictions.append(pred)
        return np.stack(predictions)

    def top_words(self, top_n: Optional[int] = 10) -> List[Dict[str, int]]:
        """
        Calculates the top words for each cluster.

        Parameters
        ----------
        top_n: int or None, default 10
            Top N words to return. If None, all words are returned.

        Returns
        -------
        list of dict of str to int
            Dictionary for each cluster mapping the words to number of occurances.
        """
        if self.vectorizer is None:
            raise NotFittedException("MovieGroupProcess: Model was not fitted.")
        feature_names = self.vectorizer.get_feature_names_out()
        dist: np.ndarray = self.cluster_word_distribution  # type: ignore
        res = []
        for i_cluster in range(self.n_clusters):
            top_indices = np.argsort(-dist[i_cluster])
            if top_n is not None:
                top_indices = top_indices[:top_n]  # type: ignore
            top_words = {
                feature_names[i]: dist[i_cluster, i]
                for i in top_indices
                if dist[
                    i_cluster, i
                ]  # Only return words if they are actually in the cluster
            }
            res.append(top_words)
        return res
