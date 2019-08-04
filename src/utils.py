import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def draw_label(img_path, color_code):
    img = Image.open(img_path)
    np_ar = np.array(img)
    rects = [(0, 0, np_ar.shape[1], np_ar.shape[0])]
    for x, y, w, h in rects:
        cv2.rectangle(np_ar, (x, y), (x+w, y+h), color_code, 50)
    return np_ar


def ap_at_k_per_query(np_query_labels, k=5):
    ap = 0.0
    running_positives = 0
    for idx, i in enumerate(np_query_labels[:k]):
        if i == 1:
            running_positives += 1
            ap_at_count = running_positives/(idx+1)
            ap += ap_at_count
    return ap/k


def get_preds_and_visualize(best_matches, query_gt_dict, img_dir, top_k_to_plot):
    # Create python list to store preds
    preds = []

    # For plotting, initialize the figures
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = int(top_k_to_plot/4)
    rows = int(top_k_to_plot/columns)

    for i, pic in enumerate(best_matches): 
        img_name = "{}.jpg".format(pic.split("/")[2].split(".")[0])
        color_code = None
        if img_name in query_gt_dict['positive']:
            color_code = (0, 255, 0)
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            color_code = (255, 255, 0)
            preds.append(1)
        else:
            color_code = (255, 0, 0)
            preds.append(0)
        
        if i + 1 > top_k_to_plot:
            continue
        else:
            img_path = "{}".format(os.path.join(img_dir, img_name))
            img_arr = draw_label(img_path, color_code)
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img_arr)
            
    plt.show()
    
    return preds



def get_preds(best_matches, query_gt_dict):
   
    # Create python list to store preds
    preds = []

    # Iterate through the best matches and find predictions
    for i, pic in enumerate(best_matches): 
        img_name = "{}".format(pic.split("/")[-1])
        if img_name in query_gt_dict['positive']:
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            preds.append(1)
        else:
            preds.append(0)
        
    return preds

# ar = np.array([1, 0, 1, 1, 1, 0, 0])
# print(average_precision_for_query(ar))