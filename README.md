<img align="left" width="82" height="82" src="docs/_static/icon.svg">

# tweetopic

:zap: Blazing Fast topic modelling over short texts in Python
<br>

[![PyPI version](https://badge.fury.io/py/tweetopic.svg)](https://pypi.org/project/tweetopic/)
[![pip downloads](https://img.shields.io/pypi/dm/tweetopic.svg)](https://pypi.org/project/tweetopic/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/centre-for-humanities-computing/tweetopic)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
<br>

<p align="center">
<img src="docs/_static/banner.svg" height=400 align="center">
</p>


## Features

- Fast :zap:
- Scalable :collision:
- High consistency and coherence :dart:
- High quality topics :fire:
- Easy visualization and inspection :eyes:
- Full scikit-learn compatibility :nut_and_bolt:

## 🛠 Installation

Install from PyPI:

```bash
pip install tweetopic
```

## 👩‍💻 Usage ([documentation](https://centre-for-humanities-computing.github.io/tweetopic/))

Train your a topic model on a corpus of short texts:

```python
from tweetopic import DMM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Creating a vectorizer for extracting document-term matrix from the
# text corpus.
vectorizer = CountVectorizer(min_df=15, max_df=0.1)

# Creating a Dirichlet Multinomial Mixture Model with 30 components
dmm = DMM(n_components=30, n_iterations=100, alpha=0.1, beta=0.1)

# Creating topic pipeline
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("dmm", dmm),
])
```

You may fit the model with a stream of short texts:

```python
pipeline.fit(texts)
```

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [topicwizard](https://github.com/x-tabdeveloping/topic-wizard).

Install it from PyPI:

```bash
pip install topic-wizard
```

Then visualize your topic model:

```python
import topicwizard

topicwizard.visualize(pipeline=pipeline, corpus=texts)
```

![topicwizard visualization](docs/_static/topicwizard.png)

## 🎓 References

- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-Based Approach for Short Text Clustering. _In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233–242). Association for Computing Machinery._
