import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from utils import plot_history

if __name__ == '__main__' :
    # Create query fts
    query_fts =  (torch.tensor([1, 2, 3, 4, 5, 6, 7]).float())

    # Create QUERY_FTS_DB
    QUERY_FTS_DB = torch.eye(7).float()

    # # Create similarity list
    # similarity = torch.matmul(query_fts, QUERY_FTS_DB.t())
    
    # print(similarity)
    # print(similarity.size())

    # print(torch.nn.functional.normalize(query_fts, dim=0))

    # input1 = torch.tensor([1, 2, 3, 4, 5, 6, 7]).float()
    # print(input1.size())
    # input1 = torch.unsqueeze(input1, dim=0)
    # print(input1.size())
    # input2 = torch.eye(7).float()
    # # input2[0][2] = 27.5
    # output = F.cosine_similarity(input2, input1)
    # print(output)

    # query_fts = np.array([1,2,3,4,5,6,7])
    # j = np.array((-query_fts).argsort().astype(int))
    # print(j)
    # print(query_fts[j])

    # m = torch.cat((query_fts, torch.tensor([8.0, 9.0, 10.0])), 0)
    # print(m)

    # a = np.array([1]*10).astype(np.int8)
    # b = np.array([0.01, 0.01, 0.01, 0, 0, 0, 0.67, 0.5,0.5, 0.3]).astype(np.float)

    # print(average_precision_score(y_true=a, y_score=b))
    



    