import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):

    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net


    def forward(self, anchor, positive, negative):
        # Get embedding for anchor and normalize
        anchor_embedding = F.normalize(self.embedding_net(anchor), p=2, dim=0)

        # Get embedding for positive and normalize
        positive_embedding = F.normalize(self.embedding_net(positive), p=2, dim=0)

        # Get embedding for negative and normalize
        negative_embedding = F.normalize(self.embedding_net(negative), p=2, dim=0)
        
        return anchor_embedding, positive_embedding, negative_embedding


    def get_embedding(self, x):
        return F.normalize(self.embedding_net(x), p=2, dim=0)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin


    def forward(self, anchor, positive, negative, size_average=False):
        #print(anchor.size(), positive.size(), negative.size())
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        #print(distance_positive, distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        #print(losses)
        return losses.mean() if size_average else losses.sum()
    
    
    def reduce_margin(self):
        self.margin = self.margin*0.8
    


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x