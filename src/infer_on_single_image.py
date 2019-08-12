from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import TripletLoss, TripletNet, Identity, create_embedding_net
from dataset import QueryExtractor, EmbeddingDataset
from torchvision import transforms
import torchvision.models as models
import torch
from utils import draw_label, ap_at_k_per_query, get_preds, get_preds_and_visualize, perform_pca_on_single_vector, ap_per_query
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from inference import get_query_embedding


def measure_performance(labels_dir, 
            img_dir, img_fts_dir,
            weights_file,
            subset="inference"):
    """
    Given a weights file, calculate the mean average precision over all the queries for the corresponding dataset

    Args:
        labels_dir  : Directory for ground truth labels
        img_dir     : Directory holding the images
        img_fts_dir : Directory holding the pca reduced features generated through create_db.py script
        weights_file: path of trained weights file
        subset      : train/ valid/ inference
    
    Returns:
        Mean Average Precision over all queries corresponding to the dataset
    """
    # Create Query extractor object
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset=subset)

    # Creat image database
    query_images = QUERY_EXTRACTOR.get_query_names()

    # Create paths
    query_image_paths = [os.path.join(img_dir, file) for file in query_images]

    aps = []
    # Now evaluate
    for i in query_image_paths:
        ap = inference_on_single_labelled_image_pca(query_img_file=i, labels_dir=labels_dir, img_dir=img_dir, img_fts_dir=img_fts_dir, weights_file=weights_file, plot=False)
        aps.append(ap)

    
    return np.array(aps).mean()


def inference_on_single_labelled_image_pca(query_img_file, 
                labels_dir,
                img_dir,
                img_fts_dir,
                weights_file,
                top_k=1000,
                plot=True,
                ):
    """
    Function that returns the average precision for a given query image and also plots the top 20 results

    Args:
        query_img_file  : path of query image file
        labels_dir  : Directory for ground truth labels
        img_dir     : Directory holding the images
        img_fts_dir : Directory holding the pca reduced features generated through create_db.py script
        weights_file: path of trained weights file
        top_k       : top_k values used to calculate the average precison
        plot        : if True, top 20 results are plotted

    Returns:
        Average precision for the query image file
    """
    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Create embedding network
    resnet_model = create_embedding_net()
    model = TripletNet(resnet_model)
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    model.eval()

    # Get query name
    query_img_name = query_img_file.split("/")[-1]
    query_img_path = os.path.join(img_dir, query_img_name)
   
    # Create Query extractor object
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="inference")
    
    # Create query ground truth dictionary
    query_gt_dict = QUERY_EXTRACTOR.get_query_map()[query_img_name]

    # Creat image database
    QUERY_IMAGES_FTS = [os.path.join(img_fts_dir, file) for file in sorted(os.listdir(img_fts_dir))]
    QUERY_IMAGES = [os.path.join(img_fts_dir, file) for file in sorted(os.listdir(img_dir))]
    
    # Query fts
    query_fts =  get_query_embedding(model, device, query_img_file).detach().cpu().numpy()
    query_fts = perform_pca_on_single_vector(query_fts)

    # Create similarity list
    similarity = []
    for file in tqdm(QUERY_IMAGES_FTS):
        file_fts = np.squeeze(np.load(file))
        cos_sim = np.dot(query_fts, file_fts)/(np.linalg.norm(query_fts)*np.linalg.norm(file_fts))
        similarity.append(cos_sim)

    # Get best matches using similarity
    similarity = np.asarray(similarity)
    indexes = (-similarity).argsort()[:top_k]
    best_matches = [QUERY_IMAGES[index] for index in indexes]
    
    # Get preds
    if plot:
        preds = get_preds_and_visualize(best_matches, query_gt_dict, img_dir, 20)
    else:
        preds = get_preds(best_matches, query_gt_dict)
    
    # Get average precision
    ap = ap_per_query(best_matches, query_gt_dict)
    
    return ap


# if __name__ == '__main__':
    # measure_performance(labels_dir="./data/oxbuild/gt_files/", img_dir="./data/oxbuild/images/", img_fts_dir="./fts_pca/oxbuild/", weights_file="./weights/oxbuild-exp-3.pth")
    # inference_on_single_labelled_image_pca(query_img_file="./data/oxbuild/images/all_souls_000013.jpg", 
    #                                         labels_dir="./data/oxbuild/gt_files/", 
    #                                         img_dir="./data/oxbuild/images/",
    #                                         img_fts_dir="./fts_pca/oxbuild/",
    #                                         top_k=500,
    #                                         plot=True,
    #                                         weights_file="./weights/oxbuild-exp-3.pth")