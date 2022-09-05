![Logo with text](./docs/_static/icon_with_text.svg)

# tweetopic: Blazing Fast Topic modelling for Short Texts

[![PyPI version](https://badge.fury.io/py/tweettopics.svg)](https://pypi.org/project/tweettopics/)
[![pip downloads](https://img.shields.io/pypi/dm/tweettopics.svg)](https://pypi.org/project/tweettopics/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/centre-for-humanities-computing/tweettopics)
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

## 🛠 Installation

The package might be released on PIP in the future, for now you can install it directly from Github with the following command:

```bash
pip install tweetopic
```

## 👩‍💻 Usage

To create a model you should import `MovieGroupProcess` from the package:

```python
from tweetopic import MovieGroupProcess

# Creating a model with 30 clusters
mgp = MovieGroupProcess(n_clusters=30, alpha=0.1, beta=0.1)
```

You may fit the model with a stream of short texts:

```python
mgp.fit(
    texts,
    n_iterations=1000, # You may specify the number of iterations
    max_df=0.1, # As well as parameters for the vectorizer.
    min_df=15
)
```

To examine the structure of the clusters you can either look at the most frequently occuring words:

```python
mgp.top_words(top_n=3)
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
mgp.visualize()
```

![PyLDAvis visualization](https://github.com/centre-for-humanities-computing/tweetopic/blob/main/docs/_static/pyldavis.png)

> Note: You must install optional dependencies if you intend to use pyLDAvis

## [API reference](https://centre-for-humanities-computing.github.io/tweetopic/)

## Limitations

tweetopic is so efficient and fast, as it exploits the fact that it's only short texts, we want to cluster. The number of unique terms in any document _MUST_ be less than 256.
Additionally any term in any document may at most appear 255 times.
<br>As of now, this requirement is not enforced by the package, please be aware of this, as it might cause strange behaviour.

> Note: If it is so demanded, this restriction could be relaxed, make sure to file an issue, if you intend to use tweetopic for longer texts.

## Differences from the [gsdmm](https://github.com/rwalk/gsdmm) package

- _tweetopic_ is usually orders of magnitude faster than _gsdmm_ thanks to the clever use of data structures and numba accelarated loops.
  After some surface level benchmarking it seems that in many cases _tweetopic_ performs the same task about 60 times faster.
- _tweetopic_ supports texts where a term occurs multiple times. _gsdmm_ only implements single occurance terms.
- gsdmm is no longer maintained

## 🎓 References

- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-Based Approach for Short Text Clustering. _In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233–242). Association for Computing Machinery._
