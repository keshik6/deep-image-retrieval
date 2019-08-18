# Deep Image Retrieval

## Introduction
The goal of this project is deep image retrieval, that is learning an embedding (or mapping) from images to a compact latent space in which cosine similarity between two learned embeddings correspond to a ranking measure for image retrieval task.

# Data

We used two popular Image retrieval datasets published by the Oxford Visual Geometry Group for this project,
1.  Oxford Buildings dataset
2.  Paris Buildings dataset

Both datasets consists of images collected from Flickr by searching for particular landmarks. The collection has been manually annotated to generate a comprehensive ground truth for all 11 different landmarks per dataset, each represented by 5 queries. This gives a set of 55 queries over which an object retrieval system can be evaluated.

We used an 80/ 20 ratio for splitting positive examples for every query to create our training and validation. The data statistics are provided below. We will discuss about the triplets in the upcoming sections.

| Dataset | #Images | #Queries | #Training Triplets | #Validation Triplets |
|---------|---------|----------|--------------------|----------------------|
| Oxford  | 5042    | 55       | 3373               | 831                  |
| Paris   | 6412    | 55       | 13230              | 3421                 |

# Problem Description
As mentioned in the introduction, the problem can be formulated as follows. 
Given a dataset D<sub>n</sub> = {
(q<sub>1</sub>, p<sub>11</sub>, ,p<sub>12</sub> ,p<sub>13</sub> , … , p<sub>1m</sub>),
(q<sub>2</sub>, p<sub>21</sub>, ,p<sub>22</sub> ,p<sub>23</sub> , … , p<sub>2k</sub>),
.....,
(q<sub>n</sub>, p<sub>n1</sub>, ,p<sub>n2</sub> ,p<sub>n3</sub> , … , p<sub>nr</sub>),
}

where q<sub>x</sub> indicates the x<sup>th</sup> query image and p<sub>xk</sub> indicates the k<sup>th</sup> positive example for the query q<sub>x</sub>. Do note that the number of positive examples for each query are not the same.

Given this dataset, our goal is to learn an embedding from these images to a compact latent space where cosine similarity between two learned embeddings correspond to a ranking measure for image retrieval task.

# Methodology and Loss function
We leverage on a siamese architecture that combines three input streams with a triplet loss. We make use of triplet loss because this has shown to be more effective for ranking problems. 

To formally describe, triplet loss is a loss function where a baseline (anchor, in our case the query image) is compared to a positive (as per annotation) image and a negative image. The triplet loss minimizes the distance from the anchor image to the positive image and maximizes the distance from the anchor image to the negative image over all the triplets in the dataset.  It is formally described below.

![alt text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/triplet_loss.png)
