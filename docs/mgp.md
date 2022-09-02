# MovieGroupProcess

<span style="color:DarkGoldenRod">class</span> **MovieGroupProcess**(n_clusters: <span style="color:cadetblue">int</span> = 8, alpha: <span style="color:cadetblue">float</span> = 0.1, beta: <span style="color:cadetblue">float</span> = 0.1, multiple_occurance: <span style="color:cadetblue">bool</span>=True)

Class for fitting a dirichlet mixture model with the movie group process algorithm described in [Yin & Wang's paper (2014)](https://dl.acm.org/doi/10.1145/2623330.2623715).

[Source](https://github.com/centre-for-humanities-computing/tweetopic/blob/main/tweetopic/mgp.py)

## **_Hyperparameters_**

**n_clusters**: <span style="color:cadetblue">int</span>, _default 8_ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of mixture components in the model.
<br>**alpha**: <span style="color:cadetblue">float</span>, _default 0.1_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Willingness of a document joining an empty cluster.
<br>**beta**: <span style="color:cadetblue">float</span>, _default 0.1_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Willingness to join clusters, where the terms in the document
are not present.
<br>**multiple_occurance**: <span style="color:cadetblue">bool</span>, _default True_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Specifies whether a term should only be counted once in each document.
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If set to False, Formula 3 will be used, else Formula 4 will be used.

## **_Attributes_**

**cluster_word_distribution**: <span style="color:cadetblue">matrix</span> _of shape (n_clusters, n_vocab)_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contains the amount a word occurs in a certain cluster.
<br>**cluster_doc_count**: <span style="color:cadetblue">array</span> _of shape (n_clusters,)_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Array containing how many documents there are in each cluster.
<br>**cluster_word_count**: <span style="color:cadetblue">array</span> _of shape (n_clusters,)_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contains the amount of words there are in each cluster.
<br>**vectorizer**: <span style="color:cadetblue">CountVectorizer</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BOW vectorizer from sklearn.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It is used to transform documents into bag of words embeddings.
<br>**n_vocab**: <span style="color:cadetblue">int</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of total vocabulary items seen during fitting.
<br>**n_documents**: <span style="color:cadetblue">int</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Total number of documents seen during fitting.

## **_Methods_**

<br>

<span style="color:DarkGoldenRod">def</span> **fit**(documents: <span style="color:cadetblue">Iterable[str]</span>, n_iterations: <span style="color:cadetblue">int</span> = 30, \*\*vectorizer_kwargs) -> <span style="color:cadetblue">MovieGroupProcess</span>

Fits the model with the MGP algorithm described in Yin and Wang (2014).

**_Parameters_**

**documents**: <span style="color:cadetblue">iterable of str</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stream of documents to fit the model with.
<br>**n_iterations**: <span style="color:cadetblue">int</span>, _default 30_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of iterations used for fitting the model.
Results usually improve with higher number of iterations.
<br>\***\*vectorizer_kwargs**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The rest of the arguments supplied are passed to sklearn's CountVectorizer.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For a detailed list of arguments consult the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

**_Returns_**<br>
<span style="color:cadetblue">MovieGroupProcess</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The same model fitted.

<br>

<br>

<span style="color:DarkGoldenRod">def</span> **transform**(embeddings: <span style="color:cadetblue">scipy.sparse.csr_matrix</span>) -> <span style="color:cadetblue">np.ndarray</span>

Predicts mixture component labels from BOW representations
of the provided documents produced by self.vectorizer.
This function is mostly here for sklearn compatibility.

**_Parameters_**

**embedings**: <span style="color:cadetblue">sparse array</span> _of shape (n_documents, n_vocab)_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BOW embeddings of documents

**_Returns_**<br>
<span style="color:cadetblue">array</span> _of shape (n_documents, n_clusters)_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Matrix of probabilities of documents belonging to each cluster.

<br>

<br>

<span style="color:DarkGoldenRod">def</span> **predict**(documents: <span style="color:cadetblue">Iterable[str]</span>) -> <span style="color:cadetblue">np.ndarray</span>

Predicts mixture component labels for the given documents.

**_Parameters_**

**documents**: <span style="color:cadetblue">iterable of str</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stream of text documents.

**_Returns_**<br>
<span style="color:cadetblue">array</span> _of shape (n_documents, n_clusters)_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Matrix of probabilities of documents belonging to each cluster.

<br>

<br>

<span style="color:DarkGoldenRod">def</span> **top_words**(top_n: <span style="color:cadetblue">int</span>) -> <span style="color:cadetblue">List[Dict[str, int]]</span>

Calculates the top words for each cluster.

**_Parameters_**

**top_n**: <span style="color:cadetblue">int</span>, _default 10_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Top N words to return. If None, all words are returned.

**_Returns_**<br>
<span style="color:cadetblue">list of dict of str to int</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary for each cluster mapping the words to number of occurances.

<br>

<br>

<span style="color:DarkGoldenRod">def</span> **most_important_words**(top_n: <span style="color:cadetblue">int</span>) -> <span style="color:cadetblue">List[Dict[str, int]]</span>

Calculates the most important words for each cluster, where
<br>`importance = n_occurances_in_component/(log(n_occurances_in_corpus) + 1)`

**_Parameters_**

**top_n**: <span style="color:cadetblue">int</span>, _default 10_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Top N words to return. If None, all words are returned.

**_Returns_**<br>
<span style="color:cadetblue">list of dict of str to int</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary for each cluster mapping the words to number of occurances.

<br>

<br>

<span style="color:DarkGoldenRod">def</span> **visualize**()

Visualizes the model with pyLDAvis for inspection of the different
mixture components :)

**_Returns_**<br>
<span style="color:cadetblue">pyLDAvis graph</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Interactive visualization of the clusters