from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch

def get_query_embedding(model, device, query_img_file):
    """
    Given a query image file path run through the model and return the embeddings

    Args:
        model           : model instance
        device          : cuda or cpu
        query_img_file  : location of the query image file

    Returns:
        The resulting embeddings
    """
    model.eval()

    # Read image
    image = Image.open(query_img_file).convert("RGB")

    # Create transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_test = transforms.Compose([transforms.Resize(460),
                                        transforms.FiveCrop(448),                               
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                        ])

   
    image = transforms_test(image)

    # Predict
    with torch.no_grad():
        # Move image to device and get crops
        image = image.to(device)
        ncrops, c, h, w = image.size()

        # Get output
        output = model.get_embedding(image.view(-1, c, h, w))
        output = output.view(ncrops, -1).mean(0)

        return output