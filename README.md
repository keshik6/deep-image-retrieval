# Deep Image Retrieval

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Problem Description](#problem-description)
- [Loss function](#loss-function)
- [How to choose triplets?](#how-to-choose-triplets)
- [Methodology](#methodology)
- [Deep Learning Architecture](#deep-learning-architecture)
- [Metrics](#metrics)
- [Hyper-parameters](#hyper-parameters)
- [Data Augmentations](#data-augmentations)
- [Dimension Reduction and deployment](#dimension-reduction-and-deployment)
- [Challenges](#challenges)
- [Results](#results)
  * [Training History](#training-history)
    - [Oxford Dataset](#oxford-dataset)
    - [Paris Dataset](#paris-dataset)
  * [Performance](#performance)
- [Flask Application for Inference](#flask-application-for-inference)
- [Google drive link](#google-drive-link)
- [How to reproduce the code?](#how-to-reproduce-the-code)
  * [Pytorch source code (Included in /src)](#pytorch-source-code)
  * [Flask App (Included in /flask_app)](#flask-app)


## Introduction
The goal of this project is deep image retrieval, that is learning an embedding (or mapping) from images to a compact latent space in which cosine similarity between two learned embeddings correspond to a ranking measure for image retrieval task. **This repository contains a simple starter implementation for deep image retrieval. Advanced features like query expansion or attention are not implemented here.**

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/demo.gif)

## Data

We used two popular Image retrieval datasets published by the Oxford Visual Geometry Group for this project,
1.  Oxford Buildings dataset
2.  Paris Buildings dataset

Both datasets consists of images collected from Flickr by searching for particular landmarks. The collection has been manually annotated to generate a comprehensive ground truth for all 11 different landmarks per dataset, each represented by 5 queries. This gives a set of 55 queries over which an object retrieval system can be evaluated.

We used an 80/ 20 ratio for splitting positive examples for every query to create our training and validation. The data statistics are provided below. We will discuss about the triplets in the upcoming sections.

| Dataset | #Images | #Queries | #Training Triplets | #Validation Triplets |
|---------|---------|----------|--------------------|----------------------|
| Oxford  | 5042    | 55       | 3373               | 831                  |
| Paris   | 6412    | 55       | 13230              | 3421                 |

## Problem Description
As mentioned in the introduction, the problem can be formulated as follows. 
Given a dataset D<sub>n</sub> = {
(q<sub>1</sub>, p<sub>11</sub>, ,p<sub>12</sub> ,p<sub>13</sub> , … , p<sub>1m</sub>),
(q<sub>2</sub>, p<sub>21</sub>, ,p<sub>22</sub> ,p<sub>23</sub> , … , p<sub>2k</sub>),
.....,
(q<sub>n</sub>, p<sub>n1</sub>, ,p<sub>n2</sub> ,p<sub>n3</sub> , … , p<sub>nr</sub>),
}

where q<sub>x</sub> indicates the x<sup>th</sup> query image and p<sub>xk</sub> indicates the k<sup>th</sup> positive example for the query q<sub>x</sub>. Do note that the number of positive examples for each query are not the same.

Given this dataset, our goal is to learn an embedding from these images to a compact latent space where cosine similarity between two learned embeddings correspond to a ranking measure for image retrieval task.

## Loss function
We leverage on a siamese architecture that combines three input streams with a triplet loss. We make use of triplet loss because this has shown to be more effective for ranking problems. 

To formally describe, triplet loss is a loss function where a baseline (anchor, in our case the query image) is compared to a positive (as per annotation) image and a negative image. The triplet loss minimizes the distance from the anchor image to the positive image and maximizes the distance from the anchor image to the negative image over all the triplets in the dataset.  It is formally described below.

![alt text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/triplet_loss.png)

Where  f<sub>i</sub><sup>a</sup>, f<sub>i</sub><sup>p</sup> and f<sub>i</sub><sup>n</sup> corresponds to the i<sup>th</sup> anchor, positive and negative embeddings respectively. We use a margin $\alpha$ to separate the embeddings.

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/triplet_learning.png)

Do note that training is quite expensive due to the fact that optimization is directly performed on the triplet space, where the number of possible triplets for training is cubic in the number of training examples.

## How to choose triplets?
A major problem with training triplet optimization problems lies in how the triplets are being chosen. For this specific problem, since we are not given any negative examples for the query, many attempts tend to choose negative examples (that excludes anchor and positive examples) randomly from the dataset and form triplets to be trained on. While this is a reasonable method, we need to show semi-hard examples to the algorithm so that it learns some quantifiable information through parametrization.
Consider the negative examples randomly sampled for the following anchor image.

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/all_souls_000051.jpg)

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/neg_ex1.jpg)

As you can clearly see, a large portion of images chosen to be negative examples are too easy, meaning that the algorithm doesn’t need to make any effort to learn to discriminate between the positive and negative examples.

We thought about both the data and the problem of choosing triplets quite carefully and decided to choose the negative images that have the highest structural similarity against the anchor as our negative examples when creating triplets.

**What is structural similarity?**  
Structural similarity measures the perceptual difference between two images. It considers image degradation as perceived change in structural information. The SSIM formula is a weighted sum based three comparison measurements between the 2 images, namely, luminance, contrast and structure. See appendix section for references.

## Methodology
Given an anchor image, we consider 500x500 center crop of anchor image against all the other non-positive images in the dataset, center cropped to 500x500 and measure the structural similarity. We select the top 500 images with the largest structural similarity as our negative example pool. 

Given this methodology, consider the hard-negatives chosen for the same query image.

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/neg_ssim_ex1.jpg)

As you can see, these examples are hard-negative examples that can allow our algorithm to learn better embeddings. In terms of implementation, we processed the query images to select top 500 negative images based on structural similarity offline and these are annotated as ‘bad’ files.

## Deep Learning Architecture
Deep neural networks have proven to be good feature extractors in the recent time since they carry out representation learning as well without any hand-engineered features. Hence, we decided to use a Resnet50 backbone as our feature extractor network where we removed the Global Average pooling layer and the fully connected layer. An example of the architecture is shown below.

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/triplet_network.png)

## Metrics
We use mean average precision over all the queries as our metric. We used the easy evaluation metric where we treated all labelled images to be positive and sampled negative images using structural similarity to be negative.

## Hyper-parameters
|              | Oxford                                                | Paris                                                 |
|--------------|-------------------------------------------------------|-------------------------------------------------------|
| Image size   | (3, 448, 448)                                         | (3, 448, 448)                                         |
| Batch size   | 64 (Parameters updated for every 64 samples)          | 64 (Parameters updated for every 64 samples)          |
| Initial lr   | 2.5e-6                                                | 5e-6                                                  |
| Optimizer    | Adam                                                  | Adam                                                  |
| Epoch        | 35                                                    | 25                                                    |
| Weight decay | 1e-5                                                  | 1e-5                                                  |
| lr scheduler | Cosine Annealing learning rate scheduler with Tmax=10 | Cosine Annealing learning rate scheduler with Tmax=10 |

## Data Augmentations 
We utilized standard augmentations including horizontal flipping, rotations, brightness adjustment, zooming, grayscaling and random resized cropping for training.

## Dimension Reduction and deployment
The embedding that we train is very high dimensional vector. So it doesn’t allow scalability when the size of your database increases. So we used Principal Component Analysis (PCA) to reduce the dimensions of the vector to 4096-dimensional vector with whitening so that our model can perform faster in real-time deployment.

Hence at test time, we run the image through our model, get the embedding, reduce using PCA to 4096-dimensional vector and use cosine similarity to obtain the most similar images from the database. 

## Challenges
Hyperparameter search was an interesting challenge we faced. Initially, our model faced overfitting problems and the model also got stuck in suboptimal local minima. We were able to resolve these issues by,
1. Using smaller learning rate so that we do not disrupt imagenet weights drastically.
2. Using learning rate scheduler rather than using static learning rates.
3. Choosing good set of image augmentations to add a small amount of noise during training to make the model robust.
4. We had to experiment with different margin rates for Triplet loss and margin of 2 seemed to work very well.

We also had to visualize training results constantly to see how our model performs visually. 

## Results
### Training History
#### Oxford Dataset
![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/oxford_loss.png)

#### Paris Dataset
![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/paris_loss.png)

### Performance 
Mean average precision and AP@k for trained models are given below. We also included the performance of MAC (Maximum Activation of Convolutions) method and Net-VLAD for comparison.

![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/performance.JPG)

As you can see our models perform reasonably given the amount of computational power we had. Generally the best performing models go for very large sizes of images in the network and also design triplets using more sophisticated mechanisms such as unsupervised/ weakly supervised triplet selection.

## Flask Application for Inference 
We built a flask application to allow users to perform visual search both on the query images as well as any new image. The HTML page is rendered on the server and displayed in a browser. A user can select the target image to search for similar images which will redirect to the inference results page that contains our model prediction. If any query images from the training set is selected, we also display the ground truth results.

The flask application is very intuitive to use. Some snips from the application are shown below.
![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/flask1.png)
![alt_text](https://github.com/keshik6/deep-image-retrieval/blob/master/readme_pics/flask2.png)

## Google drive link
All the data, pre-trained weights and pca features can be found here
https://drive.google.com/open?id=1Fy8md62TW3fmnkrv0o34ix3DjwdDK3NC

## How to reproduce the code?
### Pytorch source code
**(Included in /src)**
1. Install dependencies: pip install -r requirements.txt
2. Directory structure
* /data: Download the data from google drive link provided. You can choose to download the data from VGG website but, we have already created the negative ground truth files using structural similarity and have included in google drive. Otherwise the script will automatically start to create the negative examples which might take about 2 hrs. So using google drive to download data is recommended.
* /weights: store model weights here (pre-trained weights can be downloaded from google drive) 
* /src: contains the source code (Included in submission)
* /fts_pca: contains the pca features generated using trained networks for both datasets. (Can be downloaded from google drive)
* /results: You can store the results here (You need to create this manually)
3. Run the main function in main.py with required arguments. The codebase is clearly documented with clear details on how to execute the functions. It also includes an example. You need to interface only with this function to run the training.
4. To create the pca embeddings using your own model, use create_db.py. The function is clearly documented with an examples as well.
5. To run inference on query image files, use inference_on_single_image.py. The function is clearly documented.


### Flask App
**(Included in /flask_app). The flask application is intended to be run independently. But to avoid confusions I have included the source code for the flask app in the same repository. So do not get confused.**
1. Install dependencies : pip install -r flask_app/requirements.txt
2. Download data.zip, fts_pca.zip and weights.zip from google drive link provided above
3. Extract the downloaded folders (data, fts_pca and weights) and place them in flask_app/static/
4. Deploy the app : python flask_app/deploy.py
5. Open a web browser and go to http://localhost:5000

The inference results are stored in flask_app/static/temp folder with a unique identifier. 
Users have to manually clear the flask_app/static/temp folder if it is taking up a lot of space.

