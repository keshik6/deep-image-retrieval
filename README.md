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
