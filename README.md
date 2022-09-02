# TweeTopic

:zap: Blazing Fast implementation of the Gibbs Sampling Dirichlet Mixture Model for topic modelling over short texts utilizing the power of :1234: Numpy and :snake: Numba

## Installation

The package might be released on PIP in the future, for now you can install it directly from Github with the following command:
<br>
`pip install git+https://github.com/rwalk/gsdmm.git`

## Usage

To create a model you should import `MovieGroupProcess` from the package:

```python
from tweetopic import MovieGroupProcess

# Creating a model with 15 clusters
mgp = MovieGroupProcess(n_clusters, alpha=0.1, beta=0.1)
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

## References

Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-Based Approach for Short Text Clustering. _In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233–242). Association for Computing Machinery._
