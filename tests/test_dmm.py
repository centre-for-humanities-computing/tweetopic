import scipy.sparse as spr
from sklearn.feature_extraction.text import CountVectorizer

from tests.sample import sample_texts
from tweetopic.dmm import DMM
from tweetopic.exceptions import NotFittedException

sample = CountVectorizer().fit_transform(sample_texts)
assert isinstance(sample, spr.csr_matrix)


def test_unfitted():
    dmm = DMM()
    assert dmm.components_ is None
    assert dmm.n_documents == 0
    assert dmm.cluster_doc_count is None
    assert dmm.n_features_in_ == 0
    assert not dmm._fitted
    is_error_raised = False
    try:
        dmm._check_fitted()
    except NotFittedException:
        is_error_raised = True
    assert is_error_raised, "NotFittedException is not raised"


def test_fitting():
    dmm = DMM().fit(sample)
    sample_dense = sample.toarray()
    DMM().fit(sample_dense)
    sample_lists = sample_dense.tolist()
    DMM().fit(sample_lists)
    assert dmm._fitted
    assert dmm.n_documents == len(sample_texts)
    assert dmm.n_features_in_ != 0
    assert dmm.components_ is not None
    assert dmm.components_.shape == (dmm.n_documents, dmm.n_features_in_)
    assert dmm.cluster_doc_count is not None


def test_params():
    params = {
        "n_components": 9,
        "n_iterations": 44,
        "alpha": 0.6,
        "beta": 0.5,
    }
    dmm = DMM()
    dmm.set_params(**params)
    assert dmm.get_params() == params


def test_shapes():
    dmm = DMM().fit(sample)
    transformed = dmm.transform(sample)
    n_transformed, n_features = transformed.shape
    assert n_transformed == len(sample_texts)
    assert n_features == dmm.n_features_in_
    prediction = dmm.predict(sample)
    (n_predicted,) = prediction.shape
    assert n_predicted == len(sample_texts)
