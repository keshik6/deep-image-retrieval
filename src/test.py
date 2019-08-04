import cv2
import os
import numpy as np
import torch

if __name__ == '__main__' :
    # Create query fts
    query_fts =  torch.tensor([1, 2, 3, 4, 5, 6, 7]).float()

    # Create QUERY_FTS_DB
    QUERY_FTS_DB = torch.eye(7).float()

    # Create similarity list
    similarity = torch.matmul(query_fts, QUERY_FTS_DB.t())
    
    print(similarity)
    print(similarity.size())

    print(torch.nn.functional.normalize(query_fts, dim=0))