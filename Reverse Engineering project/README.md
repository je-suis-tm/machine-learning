# Reverse Engineering

&nbsp;
-----------------------------------------
### Table of Contents

* <a href=https://github.com/je-suis-tm/machine-learning/tree/master/Reverse%20Engineering%20project#intro>Intro</a>

* <a href=https://github.com/je-suis-tm/machine-learning/tree/master/Reverse%20Engineering%20project#centroid-model>Centroid Model</a>

* <a href=https://github.com/je-suis-tm/machine-learning/tree/master/Reverse%20Engineering%20project#distribution-model>Distribution Model</a>

* <a href=https://github.com/je-suis-tm/machine-learning/tree/master/Reverse%20Engineering%20project#density-model>Density Model</a>
------------------------------------------------
&nbsp;

### Intro

Creating a visualization from data is easy. In Tableau, it's only one click. What happens if you want to extract data from a visualization? A simple google search yields a few reverse engineering tools, yet they share the same malaise – they only work with single curve and require a lot of clicks. This project addresses these issues by incorporating unsupervised learning into image processing. Multiple curves are separated by different color channels with clustering techniques. Data can be easily extracted via computing coordinates of each pixel. A simple conversion from resolution scale to axis scale approximates the coordinates to the original spreadsheet. Voila, no more ridiculous subscription to Statista :astonished:

Well, that may be a bit exaggerating. According to the second law in thermodynamics, heat naturally flows from hotter to colder bodies unless external influence is forced upon the system. The same applies to our reverse engineering process. The irreversibility in visualization generating process determines that we would never be able to replicate the original spreadsheet. At best, we are only able to replicate the same data in the scale of the resolution. It is a consequence of the asymmetric character of image processing operations. Regardless of its limitation, it is still a great tool to capture the overall trend and rough numbers. Use it when subscription to data provider is not a commercially viable option and the requirement of the precision is off less importance.

Let’s introduce two guinea pigs of this project, a bar chart and a line chart. The bar chart comes from my <a href=https://github.com/je-suis-tm/graph-theory/blob/master/Epidemic%20Outbreak%20project>Epidemic Outbreak</a> project. It is a degree distribution of both Barabási-Albert model and Erdős-Rényi model with the similar mean degree. The real challenge lies at the overlap part of grey bars and orange bars. We need to find an algorithm to separate the grey bars from the orange bars so that we can compute the coordinates of the underlying pixels.

![Alt Text](https://github.com/je-suis-tm/graph-theory/blob/master/Epidemic%20Outbreak%20project/preview/graph-degree%20distribution.png)

Since we are only interested in the data behind the bars, we need to remove the legend, the title and the axis labels. The input should be reduced to an image fitted to the scale of the axis.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20chart.png)

The line chart comes from my <a href=https://github.com/je-suis-tm/quant-trading/tree/master/Smart%20Farmers%20project>Smart Farmer</a> project. It is the comparison between palm oil FFB forecast and palm oil CPO forward curve. The overlap of the lines is barely minimum but the tricky part is the dotted line. We need to carefully calibrate the parameters of erosion and dilation algorithm in order to preserve the inclusion of the dotted line. Without delicate balance, the dotted line could easily become noise.

![Alt Text](https://github.com/je-suis-tm/quant-trading/blob/master/Smart%20Farmers%20project/preview/oil%20palm%20vs%20palm%20oil.png)

Similar to the bar chart, the line chart has to be cleansed into a purest form without extra information.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20chart.png)

We have been taught digital images comprise of many pixels in school. The color of each pixel is represented in RGB values. Each pixel is assigned to a unique position in a 2D matrix with the same size of the resolution. Ideally, the green color in RGB is (0,255,0). A search in the matrix for all the pixels equal to (0,255,0) should yield the coordinates. The coordinates can be converted into the real data scale by multiplying the factor from resolution scale to axis scale. Palm oil CPO forward curve should be obtained. How I wish it is that easy :unamused:

The brutal truth is the green color in the line chart can range from (0,255,0) to (20,230,20). Eyes of Homo Sapiens are almost at the bottom of the food chain in contrast to infrared and ultraviolet seen by some animals. The tiny difference between (0,255,0) green and (1,255,0) green cannot be detected by us but picked up effortlessly by machines. RGB values create 2^24 combinations of colors. Here comes haute couture. How do we separate the RGB colors wisely in order to separate the multiple lines?

### Centroid Model

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20elbow%20method.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20rendered1%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20rendered2%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20chart%20re%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion1.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion2%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20chart%20re%20%20kmeans.png)

### Distribution Model

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20dirichlet.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20gmm.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20dirichlet.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20gmm.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion2%20gmm.png)

### Density Model

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20optics.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20optics.png)


