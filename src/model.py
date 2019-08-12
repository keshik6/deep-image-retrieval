import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TripletNet(nn.Module):
    """
    Class implementing the Triple Net

    Attributes:
        embedding_net: torchvision model instance. i.e. torchvision.models.resnet50(pretrained=True)
    """
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
    Class implementing Triplet loss
    It takes embeddings of an anchor sample, a positive sample and a negative sample and returns the triplet loss

    Attributes:
        margin: margin value used to seperate positive and negative samples
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
    """
    Class implementing identity module
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def create_embedding_net():
    """
    Function to create embedding net

    Returns:
        embedding net instance
    """
    # This is a resnet50 base model
    resnet_model = models.resnet50(pretrained=True)

    # Now modify the network layers
    resnet_model.fc = Identity()
    resnet_model.avgpool =  Identity()   
    #print(resnet_model)

    return resnet_model


#create_embedding_net()

