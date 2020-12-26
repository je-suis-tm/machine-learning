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

Well, that may be a bit exaggerating. According to the second law in thermodynamics, heat naturally flows from hotter to colder bodies unless external influence is forced upon the system (tenet, anyone?). The same applies to our reverse engineering process. The irreversibility in visualization generating process determines that we would never be able to replicate the original spreadsheet. At best, we are only able to replicate the same data in the scale of the resolution. It is a consequence of the asymmetric character of image processing operations. Regardless of its limitation, it is still a great tool to capture the overall trend and rough numbers. Use it when subscription to data provider is not a commercially viable option and the requirement of the precision is of less importance.

Let’s introduce two guinea pigs of this project, a bar chart and a line chart. The bar chart comes from my <a href=https://github.com/je-suis-tm/graph-theory/blob/master/Epidemic%20Outbreak%20project>Epidemic Outbreak</a> project. It is a degree distribution of both Barabási-Albert model and Erdős-Rényi model with the same mean degree. The real challenge lies at the overlap part of grey bars and orange bars. We need to find an algorithm to separate the grey bars from the orange bars so that we can compute the coordinates of the underlying pixels.

![Alt Text](https://github.com/je-suis-tm/graph-theory/blob/master/Epidemic%20Outbreak%20project/preview/graph-degree%20distribution.png)

Since we are only interested in the data “behind the bars”, we need to remove the legend, the title and the axis labels. The input should be reduced to an image fitted to the scale of the axis.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20chart.png)

The line chart comes from my <a href=https://github.com/je-suis-tm/quant-trading/tree/master/Smart%20Farmers%20project>Smart Farmer</a> project. It is the comparison between palm oil FFB forecast and palm oil CPO forward curve. The overlap of the lines is no concern, but the tricky part is the dotted line. We need to carefully calibrate the parameters of erosion and dilation algorithm in order to preserve the inclusion of the dotted line. Without delicate balance, the dotted line could easily become noise.

![Alt Text](https://github.com/je-suis-tm/quant-trading/blob/master/Smart%20Farmers%20project/preview/oil%20palm%20vs%20palm%20oil.png)

Similar to the bar chart, the line chart has to be cleansed into a purest form without extra information.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20chart.png)

We have been taught digital images comprise of many pixels in school. The color of each pixel is represented in RGB values. Each pixel is assigned to a unique position in a 2D matrix with the same size of the resolution. Ideally, the green color in RGB is (0,255,0). A search in the matrix for all the pixels equal to (0,255,0) should yield the coordinates. Palm oil CPO forward curve should be obtained. How I wish it is that easy :unamused:

The brutal truth is the green color in the line chart can range from (0,255,0) to (20,230,20) or even more dramatic. Eyes of Homo Sapiens are almost at the bottom of the food chain (I would love to see infrared and ultraviolet!). The tiny difference between (0,255,0) green and (1,255,0) green cannot be detected by our humble eyes but machines pick it up effortlessly. Yet, RGB values create 2^24 combinations of colors. Here comes haute couture. How do we separate the RGB colors wisely in order to separate the multiple lines?

### Centroid Model

Well, the answer is unsupervised learning. How? This project draws inspiration from question 5 of problem set 3 of Stanford University CS229. If you don’t bother to find it in your course materials, that’s fine. Basically, the problem set attempts to apply <a href=https://github.com/je-suis-tm/machine-learning/blob/master/k%20means.ipynb>K Means</a> to lossy image compression, by reducing the number of colors used in an image. Each pixel in the image is considered a data point in the RGB 3D space. We create a fixed number of clusters and replace each pixel with the closest cluster centroid. Ultimately, we end up with fixed number of colors.

In regard to our project, multiple lines are conventionally rendered in different colors. Same color with different line styles rarely occurs since it’s easy to confuse the audience. To separate multiple lines is equivalent to separate multiple colors. Translated to Lingua Franco in machine learning, we need the optimal number of clusters to substitute each pixel with the closest centroid. When words like “optimal” appear given the condition of “K Means”, things like “elbow method”, “silhouette score” and “gap statistics” should emerge. Apparently, elbow method is the prime choice for us due to its smaller time complexity. 

The image below is the decomposition of bar chart by elbow method. Regrettably, the overlapped part is identified as an extra cluster. The ideal number of clusters should be number of bars plus the background color, which is three.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20elbow%20method.png)

As the hyperparameters are always arbitrary, we have to rely on human intuition to calibrate the model. We plug in the magic number three and here we go. We keep the orange bars away from the grey bars.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20rendered1%20kmeans.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20rendered2%20kmeans.png)

Once different colors are separated, the rest of the operations become a lot easier. Images are represented in a `numpy` 2D matrix in `opencv2`. Decomposed images are filtered into black and white where the target is white and the rest is black. We can do a simple iteration to locate the row and column number of all the white elements in the matrix. Because the coordinates of these elements only reflect where they are in the image, a conversion from resolution scale to axis scale is necessary. For instance, the image has width and height of 100 but the actual X and Y axis ranges from -1 to 1. The coordinate of (10,20) should be converted into (0.2,0.4). Each number has to time the conversion factor (1-(-1))/100.

Last but not least, the data we obtain requires compression before usage. The compression works on both horizontal and vertical directions. From the vertical perspective, only the very top of the bar contains useful information. We merely fill anything below with the same color for the sake of aesthetics. Thus, only maximum Y value is kept given the same X value in the bar chart. In line chart, the median Y value is kept. From the horizontal perspective, the conversion from resolution scale to axis scale is insufficient. The size of the data must be shrunk from resolution scale to actual scale. For instance, the width of the image is 100. The actual data ranges from 1 to 10 with 10 data points. A straightforward way would be using double colon extended slices to take every tenth number. However, this approach largely ignores the nature of the cluster. The allocation of Y values may lean heavily towards the highly dense area. I propose the alternative way, which is based upon the minimum absolute error. For instance, we need to take two values from `[1.1,1.2,1.3,1.4,1.5,1.7,1.9,2.1]`. Say the expected value is `[1,2]`. `[::4]` expression yields `[1.1,1.5]`. Minimum absolute error approach yields `[1.1,1.9]` because 1.1 and 1.9 has the smallest error against `[1,2]`. Although in this particular case of bar chart, the horizontal compression is rather difficult due to the lack of visibility on the mean degree distribution of random graph models. The horizontal compression makes more sense in the line chart as you would’ve seen later.

Now we reverse engineer the visualization of mean degree distribution. As no information is lost, I can confidently say the entropy doesn’t increase at all :clap:

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/bar%20chart%20re%20kmeans.png)

Let’s formally define the process of reverse engineering bar chart. The procedure is identical as the bar chart.

* Préparation
* Décomposition
* Triangulation
* Conversion
* Compression

When it comes to the second lab rat, additional steps are taken in the preprocessing. The erosion is required to connect dotted line into a solid line. It makes triangulation of the coordinates a lot easier.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion1.png)

As usual, we would love to use elbow method to identify the optimal number of clusters. To our disappointment, elbow method never seems to work on the challenging part. As you can see in the notebook, only 3 clusters are identified and the connected dotted line gets the noise treatment. The objective of automation fails miserably. Hence, we manually input the magic number four, which is numbers of lines plus the background color. Abracadabra :stuck_out_tongue:

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20kmeans.png)

If you check the bottom right carefully, you’d notice the large amount of noise coinciding with the connected dotted line. They seem to be the edge color of red and green lines. Needless to say, erosion has to be performed to cleanse the decomposed image.

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion2%20kmeans.png)

The rest of the operation is the same as the bar chart. In this case, the horizontal compression is performed on the line chart and increases the entropy. The reconstructed line chart demonstrates a tiny bit of difference in the slope, although almost unrecognizable by human eyes. 

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20chart%20re%20%20kmeans.png)

### Distribution Model

Theoretically, K Means is a special case of expectation maximization algorithm. The computation of the new centroid is considered as E step. The label update of the underlying data points is considered as M step. In this chapter, we intend to examine the generalized case. One of the most common models powered by EM algorithm is <a href=https://github.com/je-suis-tm/machine-learning/blob/master/gaussian%20mixture%20model.ipynb>Gaussian Mixture Model</a>. GMM studies the mixture of different Gaussian distributions. Does GMM provide a better solution to the overlapping problem in bar chart?

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20dirichlet.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20gmm.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20dirichlet.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20gmm.jpg)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/line%20erosion2%20gmm.png)

### Density Model

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20bar%20optics.png)

![Alt Text](https://github.com/je-suis-tm/machine-learning/blob/master/Reverse%20Engineering%20project/preview/color%20channels%20line%20optics.png)



