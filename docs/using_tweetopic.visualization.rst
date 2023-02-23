.. _usage visualization:

Visualization
=============

For visualizing your topic models we recommend that you use `topicwizard <https://github.com/x-tabdeveloping/topic-wizard>`_ ,
which natively works with all sklearn-compatible topic models.

Install topicwizard from PyPI:

.. code-block:: bash

   pip install topic-wizard


topicwizard can then visualize either your :ref:`Pipeline <usage pipeline>` or your individual model components.

.. code-block:: python

   import topicwizard

   # Passing the whole pipeline
   topicwizard.visualize(pipeline=pipeline, corpus=texts)

   # Passing the individual components
   topicwizard.visualize(vectorizer=vectorizer, topic_model=dmm, corpus=texts)


.. image:: _static/topicwizard.png
   :width: 800
   :alt: Topic visualization with topicwizard

For more information consult the `documentation <https://x-tabdeveloping.github.io/topic-wizard/index.html>`_.
