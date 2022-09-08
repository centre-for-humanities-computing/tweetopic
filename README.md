![Logo with text](./docs/_static/icon_w_title.png)

# tweetopic: Blazing Fast Topic modelling for Short Texts

[![PyPI version](https://badge.fury.io/py/tweettopics.svg)](https://pypi.org/project/tweetopic/)
[![pip downloads](https://img.shields.io/pypi/dm/tweettopics.svg)](https://pypi.org/project/tweetopic/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/centre-for-humanities-computing/tweetopic)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
<br>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

:zap: Blazing Fast implementation of the Gibbs Sampling Dirichlet Mixture Model for topic modelling over short texts utilizing the power of :1234: Numpy and :snake: Numba.
<br>The package uses the Movie Group Process algorithm described in Yin and Wang (2014).

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

If you intend to use the visualization features of PyLDAvis, install the package with optional dependencies:

```bash
pip install tweetopic[viz]
```

## 👩‍💻 Usage ([documentation](https://centre-for-humanities-computing.github.io/tweetopic/))

For easy topic modelling, tweetopic provides you the TopicPipeline class:

```python
from tweetopic import TopicPipeline, DMM
from sklearn.feature_extraction.text import CountVectorizer

# Creating a vectorizer for extracting document-term matrix from the
# text corpus.
vectorizer = CountVectorizer(min_df=15, max_df=0.1)

# Creating a Dirichlet Multinomial Mixture Model with 30 components
dmm = DMM(n_clusters=30, n_iterations=100, alpha=0.1, beta=0.1)

# Creating topic pipeline
pipeline = TopicPipeline(vectorizer, dmm)
```

You may fit the model with a stream of short texts:

```python
pipeline.fit(texts)
```

To examine the structure of the topics you can either look at the most frequently occuring words:

```python
pipeline.top_words(top_n=3)
-----------------------------------------------------------------

[
    {'vaccine': 1011.0, 'coronavirus': 428.0, 'vaccines': 396.0},
    {'afghanistan': 586.0, 'taliban': 509.0, 'says': 464.0},
    {'man': 362.0, 'prison': 310.0, 'year': 288.0},
    {'police': 567.0, 'floyd': 444.0, 'trial': 393.0},
    {'media': 331.0, 'twitter': 321.0, 'facebook': 306.0},
    ...
    {'pandemic': 432.0, 'year': 427.0, 'new': 422.0},
    {'election': 759.0, 'trump': 573.0, 'republican': 527.0},
    {'women': 91.0, 'heard': 84.0, 'depp': 76.0}
]
```

Or use rich visualizations provided by [pyLDAvis](https://github.com/bmabey/pyLDAvis):

```python
pipeline.visualize(texts)
```

![PyLDAvis visualization](https://github.com/centre-for-humanities-computing/tweetopic/blob/main/docs/_static/pyldavis.png)

> Note: You must install optional dependencies if you intend to use pyLDAvis

## 🎓 References

- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-Based Approach for Short Text Clustering. _In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233–242). Association for Computing Machinery._
