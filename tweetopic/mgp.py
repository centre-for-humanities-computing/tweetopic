"""Module containing tools for fitting a Dirichlet Mixture Model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import scipy.sparse as spr
from numba import njit
from sklearn.feature_extraction.text import CountVectorizer

from tweetopic._doc import add_doc, init_doc_words, remove_doc
from tweetopic._prob import predict_doc, sample_categorical
from tweetopic.exceptions import NotFittedException


@njit
def _init_clusters(
    cluster_word_distribution: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_doc_count: np.ndarray,
    doc_clusters: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
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
        )


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
        print(f"    Transferred {total_transfers} documents.")


class MovieGroupProcess:
    """Class for fitting a dirichlet mixture model with the movie group process
    algorithm described in Yin & Wang's paper (2014).

    Parameters
    ----------
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

    def __init__(
        self,
        n_clusters: int = 8,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.cluster_word_distribution: np.ndarray | None = None
        self.cluster_doc_count: np.ndarray | None = None
        self.cluster_word_count: np.ndarray | None = None
        self.vectorizer: CountVectorizer | None = None
        self.n_documents = 0
        self.n_vocab = 0
        self._doc_term_matrix = None

    @classmethod
    def _from_attrs(cls, **attributes) -> MovieGroupProcess:
        """Creates model from a dict of its attributes.
        Exists for easier model persistence.
        """
        mgp = cls(
            n_clusters=attributes["n_clusters"],
            alpha=attributes["alpha"],
            beta=attributes["beta"],
        )
        for key, value in attributes.items():
            mgp.__setattr__(key, value)
        return mgp

    def fit(
        self,
        documents: Iterable[str],
        n_iterations: int = 30,
        **vectorizer_kwargs,
    ) -> MovieGroupProcess:
        """Fits the model with the MGP algorithm described in Yin and Wang
        (2014).

        Parameters
        ----------
        documents: iterable of str
            Stream of documents to fit the model with.
        n_iterations: int, default 30
            Number of iterations used for fitting the model.
            Results usually improve with higher number of iterations.
        **vectorizer_kwargs
            The rest of the arguments supplied are passed to sklearn's
            CountVectorizer.
            For a detailed list of arguments consult the documentation:
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        Returns
        -------
        MovieGroupProcess
            The same model fitted.

        Note
        ----
        fit() mutates the original object too, the fitted model is returned for
        convenience.
        """
        print("Converting documents to BOW matrix.")
        self.vectorizer = CountVectorizer(**vectorizer_kwargs)
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        # Saving it for visualization later
        self._doc_term_matrix = doc_term_matrix
        self.n_documents, self.n_vocab = doc_term_matrix.shape
        print("Calculating unique words.")
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            doc_term_matrix.tolil(),
        )
        print("Initialising mixture components")
        initial_clusters = np.random.multinomial(
            1,
            np.ones(self.n_clusters) / self.n_clusters,
            size=self.n_documents,
        )
        doc_clusters = np.argmax(initial_clusters, axis=1)
        self.cluster_doc_count = np.zeros(self.n_clusters)
        self.cluster_word_distribution = np.zeros((self.n_clusters, self.n_vocab))
        self.cluster_word_count = np.zeros(self.n_clusters)
        _init_clusters(
            cluster_word_distribution=self.cluster_word_distribution,
            cluster_word_count=self.cluster_word_count,
            cluster_doc_count=self.cluster_doc_count,
            doc_clusters=doc_clusters,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
        )
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

    def transform(self, embeddings: spr.csr_matrix) -> np.ndarray:
        """Predicts mixture component labels from BOW representations of the
        provided documents produced by self.vectorizer. This function is mostly
        here for sklearn compatibility.

        Parameters
        ----------
        embeddings: sparse array of shape (n_documents, n_vocab)
            BOW embeddings of documents

        Returns
        -------
        array of shape (n_documents, n_clusters)
            Matrix of probabilities of documents belonging to each cluster.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised.
        """
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

    def predict(self, documents: Iterable[str]) -> np.ndarray:
        """Predicts mixture component labels for the given documents.

        Parameters
        ----------
        documents: iterable of str
            Stream of text documents.

        Returns
        -------
        array of shape (n_documents, n_clusters)
            Matrix of probabilities of documents belonging to each cluster.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised.
        """
        if self.vectorizer is None:
            raise NotFittedException("MovieGroupProcess: Model was not fitted.")
        embeddings = self.vectorizer.transform(documents)
        return self.transform(embeddings)

    @property
    def components_(self) -> np.ndarray:
        """Alias of cluster_word_distribution for compatibility with
        sklearn."""
        if self.vectorizer is None:
            raise NotFittedException("MovieGroupProcess: Model was not fitted.")
        return self.cluster_word_distribution  # type: ignore

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
                    i_cluster,
                    i,
                ]  # Only return words if they are actually in the cluster
            }
            res.append(top_words)
        return res

    def most_important_words(self, top_n: int | None = 10) -> list[dict[str, int]]:
        """Calculates the most important words for each cluster.

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
        if self.vectorizer is None:
            raise NotFittedException("MovieGroupProcess: Model was not fitted.")
        feature_names = self.vectorizer.get_feature_names_out()
        dist: np.ndarray = self.cluster_word_distribution  # type: ignore
        word_use = dist.sum(axis=0)
        importance = dist / (
            np.log(word_use) + 1
        )  # Dividing every term by log total number of occurances.
        res = []
        for i_cluster in range(self.n_clusters):
            top_indices = np.argsort(-importance[i_cluster])
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

    def visualize(self):
        """Visualizes the model with pyLDAvis for inspection of the different
        mixture components :)

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
        return pyLDAvis.sklearn.prepare(self, self._doc_term_matrix, self.vectorizer)

    @property
    def visualise(self):
        """Alias of visualize() for those living on this side of the
        Pacific."""
        return self.visualize

    @property
    def n_components_(self) -> int:
        """Alias of n_clusters for Sklearn compatibility."""
        return self.n_clusters

    def fit_transform(
        self, documents: Iterable[str], n_iterations: int = 30, **vectorizer_kwargs
    ) -> np.ndarray:
        """Fits the model and assigns cluster labels to each document.

        Parameters
        ----------
        documents: iterable of str
            Stream of documents to fit the model with.
        n_iterations: int, default 30
            Number of iterations used for fitting the model.
            Results usually improve with higher number of iterations.
        **vectorizer_kwargs
            The rest of the arguments supplied are passed to sklearn's
            CountVectorizer.
            For a detailed list of arguments consult the documentation:
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        Returns
        -------
        array of shape (n_documents, n_clusters)
            Matrix of probabilities of documents belonging to each cluster.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised.

        Note
        ----
        The `documents` argument has to be a repeatable iterable.
        Please keep this in mind when using generators.
        We recommend that you use the @multigen wrapper in such cases:
        https://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
        """
        return self.fit(documents, n_iterations, **vectorizer_kwargs).predict(documents)

    @property
    def _primitive_params(self):
        """Returns the primitive parameters of the model as a dict."""
        return {
            "n_clusters": self.n_clusters,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_vocab": self.n_vocab,
            "n_documents": self.n_documents,
        }

    @property
    def _dense_params(self):
        """Returns the dense array parameters of the model as a dict."""
        return {
            "cluster_word_distribution": self.cluster_word_distribution,
            "cluster_doc_count": self.cluster_doc_count,
            "cluster_word_count": self.cluster_word_count,
        }

    def save(self, path: str) -> None:
        """Persists model to disk to the given directory.

        Parameters
        ----------
        path: str
            Path to the directory where the model should be saved.
            If the directory does not exist, it will be created.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self._primitive_params, os.path.join(path, "params.joblib"))
        for param_name, value in self._dense_params:
            np.save(os.path.join(path, f"{param_name}.npy"), value)
        spr.save_npz(os.path.join(path, "_doc_term_matrix.npz"), self._doc_term_matrix)
        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.joblib"))

    @classmethod
    def load(cls, path: str) -> MovieGroupProcess:
        """Loads model from the given directory.

        Parameters
        ----------
        path: str
            Path to the directory where the model should is saved.

        Returns
        -------
        MovieGroupProcess
            The saved model.
        """
        primitive_params = joblib.load(os.path.join("params.joblib"))
        dense_param_names = [
            "cluster_word_distribution",
            "cluster_doc_count",
            "cluster_word_count",
        ]
        dense_params = {
            param: np.load(os.path.join(path, f"{param}.npy"))
            for param in dense_param_names
        }
        attrs = {
            **primitive_params,
            **dense_params,
        }
        try:
            attrs["_doc_term_matrix"] = spr.load_npz(
                os.path.join(path, "_doc_term_matrix.npz")
            )
        except FileNotFoundError:
            print("Doc term matrix not found while loading, model visualization off.")
            attrs["_doc_term_matrix"] = None
        attrs["vectorizer"] = joblib.load(os.path.join(path, "vectorizer.joblib"))
        return MovieGroupProcess._from_attrs(**attrs)
