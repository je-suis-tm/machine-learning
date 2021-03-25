# Machine Learning

## Intro

Machine learning is so chic that every programmer even non-programmer starts to learn. After several months of online courses, everyone becomes self-proclaimed data scientist. The managers hold high hopes and deploy data scientists to machine learning this or that. In no time, people run into cul-de-sac, things don't work so well outside of the realm of iris dataset! If you have been to my other repositories like <a href=https://github.com/je-suis-tm/quant-trading>quant trading</a> or <a href=https://github.com/je-suis-tm/graph-theory>graph theory</a>, you must have seen me bashing reckless applications of machine learning. Stop selling AI snake oil! Don't get me wrong. I ain't no machine-learning-sceptic. I see great potential in machine learning but I am merely cynical to the current overstatement of artificial intelligence where it is frankly nowhere in sight. 

The most popular supervised learning has very rigid requirement in both data quality and data quantity. Reinforcement learning is a drain on existing hardware. On the contrary, unsupervised learning is something I mess around frequently. It greatly boosts my work efficiency by dimension reduction, although I struggle to interpret the substantial meaning of the clustering pattern from time to time. In short, machine learning is no panacea. Its strongest suit is classification with discrete answers. When it comes to predicting stock price tomorrow or computing basic reproduction number yesterday, we still have to take the conventional path. 

This repository is based upon the <a href=http://cs229.stanford.edu/syllabus-fall2020.html>course material</a> by Stanford University. Professor Andrew Ng may not teach the most comprehensive lectures but he has inspired millions to study data science. This repository attempts to replicate every algorithm mentioned in the course as well as the popular ones outside of the course. The experienced coders urge us not to reinvent the wheel but I firmly believe we never truly understand how a wheel works until we reinvent it. If you only learn OPTICS from some articles on towardsdatascience.com, you would've skipped DBSCAN since OPTICS does not require the key input ε. Well, by reinventing the wheels, you would come to senses that this is purely quid pro quoi. The introduction of new input ξ is crucial to determine the clustering. Yet, few people talk about it. In that sense, data modelling is not really scientific and will never be that way. Machine learning is a state of art where you fine tune the parameters to create discrete answers to the real-life problems. I sincerely hope this repository can help you see that.

<hr>

## Algorithms

### Supervised

* Generative Learning (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/gaussian%20discriminant%20analysis.ipynb>Gaussian Discriminant Analysis</a>)

* Gradient Descent (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/gradient%20descent.ipynb>Batch</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/gradient%20descent.ipynb>Stochastic</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/gradient%20descent.ipynb>Mini-batch</a>)

* Naïve Bayes (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/naive%20bayes.ipynb>Multivariate</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/naive%20bayes.ipynb>Multinomial</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/naive%20bayes.ipynb>TF-IDF</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/naive%20bayes.ipynb>KL Divergence</a>)

* Newton Method (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/newton%20method%20for%20logistic%20regression.ipynb>Logistic Regression</a>)

* Support Vector Machine (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/binary%20support%20vector%20machine.ipynb>Binary</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/multiclass%20support%20vector%20machine.ipynb>Multiclass</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/multiclass%20support%20vector%20machine.ipynb>DAG</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/sequential%20minimal%20optimization.ipynb>SMO</a>)

* Tree-based Learning (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/decision%20tree%20and%20random%20forest.ipynb>Decision Tree</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/decision%20tree%20and%20random%20forest.ipynb>Random Forest</a>)

* Instance-based Learning (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/k%20nearest%20neighbors.ipynb>K Nearest Neighbors</a>)

### Unsupervised

* Centroid-based Model (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/k%20means.ipynb>K Means</a>)

* Density-based Model (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/dbscan.ipynb>DBSCAN</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/optics.ipynb>OPTICS</a>)

* Distribution-based Model (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/gaussian%20mixture%20model.ipynb>Gaussian Mixture Model</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/naive%20bayes%20mixture%20model.ipynb>Naïve Bayes Mixture Model</a>)

* Expectation Maximization (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/factor%20analysis.ipynb>Factor Analysis</a>/Dawid-Skene Model/Platt-Burges Model)

* Principal Component Analysis (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/principal%20component%20analysis.ipynb>Singular Value Decomposition</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/principal%20component%20analysis.ipynb>Low Rank Approximation</a>/<a href=https://github.com/je-suis-tm/machine-learning/blob/master/latent%20semantic%20indexing.ipynb>Latent Semantic Indexing</a>)

* Signal Processing (<a href=https://github.com/je-suis-tm/machine-learning/blob/master/independent%20component%20analysis.ipynb>Independent Component Analysis</a>)


## Applications

### 1. Reverse Engineering project

Creating a visualization from data is easy. In Tableau, it's only one click. What happens if you want to extract data from a visualization? A simple google search yields a few reverse engineering tools, yet they share the same malaise – they only work with single curve and require a lot of clicks. This project addresses these issues by incorporating unsupervised learning into image processing. Multiple curves are separated by different color channels with clustering techniques. Data can be easily extracted via computing coordinates of each pixel. A simple conversion from resolution scale to axis scale approximates the coordinates to the original spreadsheet. Voila, no more ridiculous subscription to Statista :astonished:

![alt text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20elbow%20method.png)

For more details, please refer to the <a href=https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/README.md>read me page</a> of a separate directory or <a href=https://je-suis-tm.github.io/machine-learning/reverse-engineering>machine learning section on my personal blog.
  
### 2. Wisdom of Crowds project

Every now and then, we read some bulge brackets hit the headline, “XXX will reach 99999€ in 20YY”. Some forecasts hit the bull’s eye but most projections are as accurate as astrology. Price prediction can be easily influenced by the cognitive bias. In the financial market, there is merit to the idea that <a href=https://www.investopedia.com/terms/c/consensusestimate.asp>consensus estimate</a> is the best oracle. By harnessing the power of ensemble learning, we are about to leverage Dawid-Skene model and Platt-Burges model to eliminate the idiosyncratic noise associate with each individual judgement. The end game is to reveal the underlying intrinsic value generated by the collective knowledge of research analysts from different investment banks. Is wisdom of crowds a crystal ball for trading? 

For more details, please refer to the <a href=https://github.com/je-suis-tm/machine-learning/blob/master/Wisdom%20of%20Crowds%20project/README.md>read me page</a> of a separate directory or <a href=https://je-suis-tm.github.io/machine-learning/wisdom-of-crowds>machine learning section on my personal blog.
