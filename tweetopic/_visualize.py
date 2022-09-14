"""Module for aiding visualization with pyLDAvis."""

import warnings
from typing import Union

import numpy as np
import pyLDAvis
import scipy.sparse as spr
from pyLDAvis._prepare import PreparedData
from pyLDAvis.sklearn import _row_norm

from tweetopic.typing import TopicModel, Vectorizer

try:
    pyLDAvis.enable_notebook()
except AttributeError:
    warnings.warn(
        "Could not enable notebook displaying,"
        "visualizations can only be saved, not displayed.",
    )


def prepare_pipeline(
    vectorizer: Vectorizer,
    model: TopicModel,
    embeddings: Union[np.ndarray, spr.spmatrix],
) -> PreparedData:
    """Prepares data for visualization."""
    return pyLDAvis.prepare(
        vocab=vectorizer.get_feature_names_out(),
        doc_lengths=embeddings.sum(axis=1).getA1(),
        term_frequency=embeddings.sum(axis=0).getA1(),
        topic_term_dists=_row_norm(model.components_),
        doc_topic_dists=_row_norm(model.transform(embeddings)),
        start_index=0,
        sort_topics=False,
    )
