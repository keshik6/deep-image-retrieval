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
    
    # tr_hist = [1146.9516,
    # 955.4418,
    # 865.9419,
    # 752.8061,
    # 704.5860,
    # 660.3335,
    # 588.5612,
    # 555.8679,
    # 519.0015,
    # 499.8029,
    # 467.9747,
    # 478.1490,
    # 420.1725,
    # 398.6504,
    # 404.2427,
    # 421.1928,
    # 343.1774,
    # 342.7617,
    # 313.1199,
    # 322.8611,
    # 283.6265,
    # 294.1859,
    # 263.6775,
    # 276.5064,
    # 296.6328,
    # 227.2783,
    # 209.8197,
    # 172.4703,
    # 169.7355,
    # 157.8034,
    # 148.2655,
    # 132.4338,
    # 111.3829,
    # 96.6675,
    # 87.8546]

    # val_hist = [1374.9731,
    # 1235.1793,
    # 1144.1744,
    # 1068.0036,
    # 1058.8519,
    # 1007.6832,
    # 1003.6021,
    # 992.3670,
    # 990.0371,
    # 952.3260,
    # 957.5153,
    # 960.6579,
    # 914.2211,
    # 925.3434,
    # 903.7935,
    # 897.2681,
    # 871.9420,
    # 870.1113,
    # 847.5705,
    # 843.0888,
    # 849.8536,
    # 865.1766,
    # 846.4319,
    # 844.3688,
    # 836.0172,
    # 828.5999,
    # 757.6686,
    # 743.8029,
    # 715.8288,
    # 729.3822,
    # 701.0769,
    # 680.4411,
    # 648.2611,
    # 657.4036,
    # 648.8640
    # ]

    # print(len(val_hist))

    # plot_history(tr_hist, val_hist, "Triplet Loss", "a.png")



    