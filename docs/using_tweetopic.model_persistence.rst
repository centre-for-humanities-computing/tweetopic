Model persistence
=================

For model persistence we suggest you use joblib,
since it stores numpy arrays much more efficiently:

.. code-block:: python

    from joblib import dump, load
    dump(dmm, "dmm_model.joblib")

You may load the model as follows:

.. code-block:: python

    dmm = load("dmm_model.joblib")

.. note::
    For a comprehensive overview of limitations,
    consult `Sklearn's documentation <https://scikit-learn.org/stable/model_persistence.html>`_