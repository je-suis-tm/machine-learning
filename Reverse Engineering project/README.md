# Reverse Engineering

Creating a visualization from data is easy. In Tableau, it's only one click. What happens if you want to extract data from a visualization? A simple google search yields a few reverse engineering tools, yet they share the same malaise – they only work with single curve and require a lot of clicks. This project addresses these issues by incorporating unsupervised learning into image processing. Multiple curves are separated by different color channels with clustering techniques. Data can be easily extracted via computing coordinates of each pixel. A simple conversion from resolution scale to axis scale approximates the coordinates to the original spreadsheet. Voila, no more ridiculous subscription to Statista :astonished:

### K Means

##### Bar Chart

![Alt Text](https://github.com/je-suis-tm/graph-theory/blob/master/Epidemic%20Outbreak%20project/preview/graph-degree%20distribution.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20chart.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20elbow%20method.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20rendered1%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20rendered2%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20chart%20re%20kmeans.png)

##### Line Chart

![Alt Text](https://github.com/je-suis-tm/quant-trading/blob/master/Smart%20Farmers%20project/preview/oil%20palm%20vs%20palm%20oil.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20chart.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion1.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion2%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20chart%20re%20%20kmeans.png)

### Mixture

##### Bar Chart

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20dirichlet.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20gmm.jpg)

##### Line Chart

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20dirichlet.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20gmm.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion2%20gmm.png)

### Density

##### Bar Chart

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20optics.png)

##### Line Chart

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20optics.png)
