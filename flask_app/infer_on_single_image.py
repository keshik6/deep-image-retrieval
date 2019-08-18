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
from utils import draw_label, ap_at_k_per_query, get_preds, get_gt_web, get_preds_and_visualize, perform_pca_on_single_vector, ap_per_query
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


def getQueryNames(labels_dir="./static/data/oxbuild/gt_files/", 
                img_dir="./static/data/oxbuild/images/"):

    """
    Function that returns a list of images that are part of validation set

    Args:
        labels_dir  : Directory for ground truth labels
        img_dir     : Directory holding the images

    Returns:
        List of file paths for images that are part of validation set
    """
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="inference")
    query_names=QUERY_EXTRACTOR.get_query_names()
    for i in range(len(query_names)):
        query_names[i]=img_dir[1:]+query_names[i]
    return QUERY_EXTRACTOR.get_query_names()

    

def getModel(weights_file="./static/weights/oxbuild_final.pth"):

    """
    Function that returns the model (saved during deploy stage to redce load time)

    Args:
        weights_file: path of trained weights file

    Returns:
        model based on weights_file
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    resnet_model = create_embedding_net()
    model = TripletNet(resnet_model)
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    model.eval()
    return model

def inference_on_single_labelled_image_pca_web(model, query_img_file, 
                labels_dir="./static/data/oxbuild/gt_files/", 
                img_dir="./static/data/oxbuild/images/",
                img_fts_dir="./static/fts_pca/oxbuild/",
                top_k=60,
                plot=False,
                ):
    
    """
    Function similar to inference_on_single_labelled_image_pca, but modified return values for usage during deployment

    Args:
        model       : model used (either paris or oxford)
        query_img_file  : path of query image file
        labels_dir  : Directory for ground truth labels
        img_dir     : Directory holding the images
        img_fts_dir : Directory holding the pca reduced features generated through create_db.py script
        top_k       : top_k values used to calculate the average precison, default is 60 for web deployment
        plot        : if True, top 20 results are plotted

    Returns:
        List of top k similar images; list of ground truth labels for top k images
    """
    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Get query name
    query_img_name = query_img_file.split("/")[-1]
    query_img_path = os.path.join(img_dir, query_img_name)
   
    # Create Query extractor object
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="inference")

    # Creat image database
    QUERY_IMAGES_FTS = [os.path.join(img_fts_dir, file) for file in sorted(os.listdir(img_fts_dir))]
    QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))]
    
    # Query fts
    query_fts =  get_query_embedding(model, device, "."+query_img_file).detach().cpu().numpy()
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
    print(best_matches)
    
    # Create query ground truth dictionary
    gt_map=[0]*60
    try:
        query_gt_dict = QUERY_EXTRACTOR.get_query_map()[query_img_name]
        gt_map=get_gt_web(best_matches,query_gt_dict)
    except:
        pass
    print(gt_map)
    for i in range(len(best_matches)):
        best_matches[i]=best_matches[i][1:]
    return best_matches, gt_map

def inference_on_single_labelled_image_pca_web_original(model, query_img_file, 
                img_dir="./static/data/oxbuild/images/",
                img_fts_dir="./static/fts_pca/oxbuild/",
                img_dir2="./static/data/paris/images/",
                img_fts_dir2="./static/fts_pca/paris/",
                top_k=60,
                plot=False,
                ):
    
    """
    Function similar to inference_on_single_labelled_image_pca, but modified return values for usage during web deployment for original images where ground truths are unavailable

    Args:
        model       : model used (either paris or oxford)
        query_img_file  : path of query image file
        img_dir     : Directory holding the images (oxford)
        img_fts_dir : Directory holding the pca reduced features generated through create_db.py script (oxford)
        img_dir2    : Directory holding the images (paris)
        img_fts_dir2: Directory holding the pca reduced features generated through create_db.py script (paris)
        top_k       : top_k values used to calculate the average precison, default is 60 for web deployment
        plot        : if True, top 20 results are plotted

    Returns:
        List of top k similar images
    """

    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Creat image database
    QUERY_IMAGES_FTS = [os.path.join(img_fts_dir, file) for file in sorted(os.listdir(img_fts_dir))] + [os.path.join(img_fts_dir2, file) for file in sorted(os.listdir(img_fts_dir2))]
    QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))] + [os.path.join(img_dir2, file) for file in sorted(os.listdir(img_dir2))]
    
    # Query fts
    query_fts =  get_query_embedding(model, device, "."+query_img_file).detach().cpu().numpy()
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
    #print(best_matches)
    for i in range(len(best_matches)):
        best_matches[i]=best_matches[i][1:]
    return best_matches

if __name__ == '__main__':
    #validate(subset="train")
    #inference_on_single_labelled_image(query_img_file="./data/oxbuild/images/all_souls_000026.jpg", weights_file="./weights/oxbuild-exp-1.pth")
    #inference_on_single_labelled_image_pca(query_img_file="./data/oxbuild/images/christ_church_000999.jpg", weights_file="./weights/oxbuild-exp-2.pth")
    pass