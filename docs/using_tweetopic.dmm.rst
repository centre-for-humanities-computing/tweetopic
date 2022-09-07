Dirichlet Mixture Model
=======================

tweetopic provides a fully sklearn compatible model for topic modelling.
The Dirichlet mixture model pretty much has the same usage as and NMF or LDA model from sklearn.


Creating a model:

.. code-block:: python

    from tweetopic import DMM

    dmm = DMM(
        n_components=15,
        n_iterations=200,
        alpha=0.1,
        beta=0.2,
    )

Fitting the model on a document-term matrix:

.. code-block:: python

    dmm.fit(doc_term_matrix)

Predicting cluster labels for unseen documents:

.. code-block:: python
    
    dmm.transform(new_docs)

:ref:`API reference <tweetopic dmm>`