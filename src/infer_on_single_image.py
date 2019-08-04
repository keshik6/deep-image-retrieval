from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import TripletLoss, TripletNet, Identity
from dataset import QueryExtractor, EmbeddingDataset
from torchvision import transforms
import torchvision.models as models
import torch
from utils import draw_label, ap_at_k_per_query, get_preds, get_preds_and_visualize
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from inference import get_query_embedding


def inference_on_single_labelled_image(query_img_file, 
                labels_dir="./data/oxbuild/gt_files/", 
                img_dir="./data/oxbuild/images/",
                top_k=25,
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
    resnet_model = models.resnet101(pretrained=True)
    #resnet_model.fc = Identity()
    model = TripletNet(resnet_model)
    model.load_state_dict(weights_file)
    model.to(device)

    # Get query name
    query_img_name = query_img_file.split("/")[-1]
    query_img_path = os.path.join(img_dir, query_img_name)
   
    # Create Query extractor object
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="valid")
    if query_img_name not in QUERY_EXTRACTOR.get_query_names():
        QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="train")
    
    print(QUERY_EXTRACTOR.get_query_names())
    # Create query ground truth dictionary
    query_gt_dict = QUERY_EXTRACTOR.get_query_map()[query_img_name]

    # Creat image database
    QUERY_IMAGES = [os.path.join(img_dir, file) for file in os.listdir(img_dir)]

    # Query fts
    query_fts =  get_query_embedding(model, device, query_img_path)

    # Create QUERY FTS numpy matrix
    print("> Creating feature embeddings")
    QUERY_FTS_DB = None
    eval_transforms = transforms.Compose([transforms.Resize(460),
                                        transforms.CenterCrop(448),
                                        transforms.ToTensor(),
                                        ])

    eval_dataset = EmbeddingDataset(image_dir=img_dir, query_img_list = QUERY_IMAGES, transforms=eval_transforms)
    eval_loader = DataLoader(eval_dataset, batch_size=5, num_workers=4, shuffle=False)

    with torch.no_grad():
        for idx, images in enumerate(tqdm(eval_loader)):
            images = images.to(device)
            output = model.get_embedding(images)

            if idx == 0:
                QUERY_FTS_DB = output
            else:
                QUERY_FTS_DB = torch.cat((QUERY_FTS_DB, output), 0)

            del images, output
            gc.collect()
            torch.cuda.empty_cache()


    # Evaluate on training set
    print("> Evaluate")
    
    # Create similarity list
    similarity = torch.matmul(query_fts, QUERY_FTS_DB.t())

    # Get best matches using similarity
    similarity = similarity.cpu().numpy()
    indexes = (-similarity).argsort()[:top_k]
    best_matches = [QUERY_IMAGES[index] for index in indexes]
    
    # Get preds
    preds = get_preds(best_matches, query_gt_dict)
    
    # Get average precision
    ap = ap_at_k_per_query(preds, top_k)
    
    # Get preds
    if plot:
        preds = get_preds_and_visualize(best_matches, query_gt_dict, img_dir, 20)
    else:
        preds = get_preds(best_matches, query_gt_dict, img_dir)
    
    print(ap)

    return ap


if __name__ == '__main__':
    inference_on_single_labelled_image(query_img_file="./data/oxbuild/images/all_souls_000051.jpg", weights_file="./weights/temp-triplet_model.pth")