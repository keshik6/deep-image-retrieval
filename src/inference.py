from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import TripletLoss, TripletNet, Identity
from dataset import QueryExtractor, EvalDataset
from torchvision import transforms
import torchvision.models as models
import torch
from utils import draw_label, ap_at_k_per_query, get_preds, get_preds_and_visualize
from sklearn.metrics import average_precision_score


def get_query_embedding(query_img_file, 
                        device, 
                        model_weights_path="./weights/triplet_model.pth",
                        ):
    
    # Create transformss
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_test = transforms.Compose([transforms.Resize(280),
                                        transforms.FiveCrop(256),                                 
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        ])

    # Read image
    image = Image.open(query_img_file)
    image = transforms_test(image)

    # Create embedding network
    resnet_model = models.resnet101(pretrained=False)
    model = TripletNet(resnet_model)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        # Move image to device and get crops
        image = image.to(device)
        ncrops, c, h, w = image.size()

        # Get output
        output = model.get_embedding(image.view(-1, c, h, w))
        output = output.view(ncrops, -1).mean(0).cpu().numpy()

        return output


def inference_on_single_labelled_image(query_img_file, 
                img_fts_dir="./fts/", 
                labels_dir="./data/oxbuild/gt_files/", 
                img_dir="./data/oxbuild/images/",
                top_k=100,
                plot=True,
                ):
    
    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Get query name
    query_img_name = query_img_file.split("/")[-1]

    # Create Query extractor object
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="valid")
    if query_img_name not in QUERY_EXTRACTOR.get_query_names():
        QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="train")
    
    # Create query ground truth dictionary
    query_gt_dict = QUERY_EXTRACTOR.get_query_map()[query_img_name]

    # Creat image database
    QUERY_IMAGES = [os.path.join(img_fts_dir, file) for file in os.listdir(img_fts_dir)]

    # Query fts
    query_fts =  get_query_embedding(query_img_file, device)

    # Create similarity list
    similarity = []
    for file in QUERY_IMAGES:
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
        preds = get_preds(best_matches, query_gt_dict, img_dir)
    
    # Get average precision
    ap = ap_at_k_per_query(preds, top_k)

    return ap


def inference_on_set(img_fts_dir="./fts/", 
                labels_dir="./data/oxbuild/gt_files/", 
                img_dir="./data/oxbuild/images/",
                top_k=25,
                subset="train",
                weights_path="./weights/",
                device=None,
                ):
    
    # Create Query extractor object
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset=subset)

    # Create ap list
    ap_list = []

    for query_img_name in tqdm(QUERY_EXTRACTOR.get_query_names()):
        # Create query ground truth dictionary
        query_gt_dict = QUERY_EXTRACTOR.get_query_map()[query_img_name]

        # Create query image file path
        query_img_file = os.path.join(img_dir, query_img_name)
        
        # Creat image database
        QUERY_IMAGES = [os.path.join(img_fts_dir, file) for file in os.listdir(img_fts_dir)]

        # Query fts
        query_fts =  get_query_embedding(query_img_file, weights_path, device)

        # Create similarity list
        similarity = []
        for file in QUERY_IMAGES:
            file_fts = np.squeeze(np.load(file))
            cos_sim = np.dot(query_fts, file_fts)/(np.linalg.norm(query_fts)*np.linalg.norm(file_fts))
            similarity.append(cos_sim)

        # Get best matches using similarity
        similarity = np.asarray(similarity)
        indexes = (-similarity).argsort()[:top_k]
        best_matches = [QUERY_IMAGES[index] for index in indexes]
        
        # Get preds
        preds = get_preds(best_matches, query_gt_dict, img_dir)
        
        # Get average precision
        ap = ap_at_k_per_query(preds, top_k)
        ap_list.append(ap)

    return np.array(ap_list).mean()
    

# ap = inference_on_single_labelled_image(query_img_file="./data/oxbuild/images/all_souls_000051.jpg")
# print(ap)


# ap = inference_on_set(subset="train")
# print(ap)

