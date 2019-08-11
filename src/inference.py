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
from torch.nn.functional import cosine_similarity


def get_query_embedding(model, 
                        device,
                        query_img_file,  
                        ):
    
    model.eval()

    # Read image
    image = Image.open(query_img_file)

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


# def inference_on_set(model,
#                 labels_dir="./data/oxbuild/gt_files/", 
#                 img_dir="./data/oxbuild/images/",
#                 top_k=50,
#                 device=None,
#                 ):
    
#     model.eval()

#     # Create Query extractor object
#     QUERY_EXTRACTOR_TRAIN = QueryExtractor(labels_dir, img_dir, subset="train")
#     QUERY_EXTRACTOR_VALID = QueryExtractor(labels_dir, img_dir, subset="valid")

#     # Creat image database
#     QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))]


#     # Create QUERY FTS numpy matrix
#     print("> Creating feature embeddings")
#     QUERY_FTS_DB = None
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     eval_transforms = transforms.Compose([transforms.Resize(280),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean=mean, std=std),
#                                         ])

#     # eval_dataset = EmbeddingDataset(image_dir=img_dir, query_img_list = QUERY_IMAGES, transforms=eval_transforms)
#     # eval_loader = DataLoader(eval_dataset, batch_size=12, num_workers=4, shuffle=False)
    
#     # print(QUERY_IMAGES)
#     # with torch.no_grad():
#     #     for idx, images in enumerate(tqdm(eval_loader)):
#     #         images = images.to(device)
#     #         output = model.get_embedding(images)

#     #         if idx == 0:
#     #             QUERY_FTS_DB = output
#     #         else:
#     #             QUERY_FTS_DB = torch.cat((QUERY_FTS_DB, output), 0)
            
#     #         del images, output
#     #         gc.collect()
#     #         torch.cuda.empty_cache()

#     create_embeddings_db("./weights/temp-triplet_model.pth", img_dir="./data/oxbuild/images/", fts_dir="./fts/")

#     # Create ap list
#     ap_list_train, ap_list_valid = [], []
    
#     QUERY_IMAGES_FTS = [os.path.join("./fts/", file) for file in sorted(os.listdir("./fts/"))]
    
#     # Evaluate on training set
#     print("> Calculating mAP on training set")
#     for query_img_name in tqdm(QUERY_EXTRACTOR_TRAIN.get_query_names()):
#         # Create query ground truth dictionary
#         query_gt_dict = QUERY_EXTRACTOR_TRAIN.get_query_map()[query_img_name]

#         # Create query image file path
#         query_img_file = os.path.join(img_dir, query_img_name)
        
#         # Query fts
#         query_fts =  get_query_embedding(model, device, query_img_file).detach().cpu().numpy()
#         #query_fts = torch.unsqueeze(query_fts, dim=0)

#         # # Create similarity list
#         # similarity = cosine_similarity(QUERY_FTS_DB, query_fts)

#         similarity = []
#         for file in QUERY_IMAGES_FTS:
#             file_fts = np.squeeze(np.load(file))
#             cos_sim = np.dot(query_fts, file_fts)/(np.linalg.norm(query_fts)*np.linalg.norm(file_fts))
#             similarity.append(cos_sim)

#         # Get best matches using similarity
#         # similarity = similarity.cpu().numpy()
#         similarity = np.array(similarity)
#         indexes = (-similarity).argsort()[:top_k]
#         best_matches = [QUERY_IMAGES[index] for index in indexes]

#         # Get preds
#         preds = get_preds(best_matches, query_gt_dict)
        
#         # Get average precision
#         ap = ap_at_k_per_query(preds, top_k)
#         ap_list_train.append(ap)
    

#     # Evaluate on validation set
#     print("> Calculating mAP of validation set")
#     for query_img_name in tqdm(QUERY_EXTRACTOR_VALID.get_query_names()):
#         # Create query ground truth dictionary
#         query_gt_dict = QUERY_EXTRACTOR_VALID.get_query_map()[query_img_name]

#         # Create query image file path
#         query_img_file = os.path.join(img_dir, query_img_name)
        
#         # Query fts
#         query_fts =  get_query_embedding(model, device, query_img_file).detach().cpu().numpy()
#         #query_fts = torch.unsqueeze(query_fts, dim=0)

#         # # Create similarity list
#         # similarity = cosine_similarity(QUERY_FTS_DB, query_fts)

#         similarity = []
#         for file in QUERY_IMAGES_FTS:
#             file_fts = np.squeeze(np.load(file))
#             cos_sim = np.dot(query_fts, file_fts)/(np.linalg.norm(query_fts)*np.linalg.norm(file_fts))
#             similarity.append(cos_sim)


#         # Get best matches using similarity
#         similarity = np.array(similarity)
#         indexes = (-similarity).argsort()[:top_k]
#         best_matches = [QUERY_IMAGES[index] for index in indexes]
        
#         # Get preds
#         preds = get_preds(best_matches, query_gt_dict)
        
#         # Get average precision
#         ap = ap_at_k_per_query(preds, top_k)
#         ap_list_valid.append(ap)

#     print(ap_list_train, ap_list_valid)
#     return np.array(ap_list_train).mean(), np.array(ap_list_valid).mean()
    

# ap = inference_on_single_labelled_image(query_img_file="./data/oxbuild/images/all_souls_000026.jpg", top_k=50)
# print(ap)


# ap = inference_on_set(subset="train")
# print(ap)

