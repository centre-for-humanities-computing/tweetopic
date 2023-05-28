.. _usage pipeline:

Pipelines
=========

To avoid data leakage and make it easier to operate with topic models, we recommend that you use scikit-learn's `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html>`_

Create a vectorizer and topic model:

.. code-block:: python

    from tweetopic import DMM
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.3,
        min_df=15,
    )
    dmm = DMM(
        n_components=15,
        n_iterations=200,
        alpha=0.1,
        beta=0.2,
    )

Add the two components to a tweetopic pipeline:

.. code-block:: python

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("dmm", dmm)
    ])

Fit pipelines on a stream of texts:

.. code-block:: python

    pipeline.fit(texts)

.. note::
    It is highly advisable to pre-process texts with an NLP library
    such as `Spacy <https://spacy.io/>`_ or `NLTK <https://www.nltk.org/>`_.
    Removal of stop/function words and lemmatization could drastically improve the quality of topics. 


