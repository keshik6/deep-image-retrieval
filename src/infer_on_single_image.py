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

def inference_on_single_labelled_image(query_img_file, 
                labels_dir="./data/oxbuild/gt_files/", 
                img_dir="./data/oxbuild/images/",
                img_fts_dir="./fts/",
                top_k=50,
                plot=True,
                weights_file=None,
                ):
    
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
    
    # Get preds
    if plot:
        preds = get_preds_and_visualize(best_matches, query_gt_dict, img_dir, 20)
    else:
        preds = get_preds(best_matches, query_gt_dict)
    
    # Get average precision
    # ap = ap_at_k_per_query(preds, top_k)
    
    ap = ap_per_query(best_matches, query_gt_dict)
    print(ap)

    return ap


def validate(labels_dir, 
            img_dir, img_fts_dir,
            weights_file,
            subset="inference"):
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
    print(best_matches)
    
    # Get preds
    if plot:
        preds = get_preds_and_visualize(best_matches, query_gt_dict, img_dir, 20)
    else:
        preds = get_preds(best_matches, query_gt_dict)
    
    # Get average precision
    ap = ap_per_query(best_matches, query_gt_dict)
    
    return ap


if __name__ == '__main__':
    # validate(labels_dir="./data/oxbuild/gt_files/", img_dir="./data/oxbuild/images/", img_fts_dir="./fts_pca/oxbuild/", weights_file="./weights/oxbuild-exp-3.pth")
    # inference_on_single_labelled_image(query_img_file="./data/oxbuild/images/all_souls_000026.jpg", weights_file="./weights/oxbuild-exp-1.pth")
    # inference_on_single_labelled_image_pca(query_img_file="./data/oxbuild/images/all_souls_000051.jpg", 
    #                                         labels_dir="./data/oxbuild/gt_files/", 
    #                                         img_dir="./data/oxbuild/images/",
    #                                         img_fts_dir="./fts_pca/oxbuild/",
    #                                         top_k=500,
    #                                         plot=True,
    #                                         weights_file="./weights/oxbuild-exp-3.pth")