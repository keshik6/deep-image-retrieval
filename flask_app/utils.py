import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.measure import compare_ssim
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score


def draw_label(img_path, color_code):
    """
    Function that draws a rectangle around an image given the rgb color code

    Args:
        img_path    : path of the image
        color_code  : color code of the bounding box
    
    Returns:
        numpy array with the rectangle drawn
    """
    img = Image.open(img_path)
    np_ar = np.array(img)
    rects = [(0, 0, np_ar.shape[1], np_ar.shape[0])]
    for x, y, w, h in rects:
        cv2.rectangle(np_ar, (x, y), (x+w, y+h), color_code, 50)
    return np_ar


def ap_at_k_per_query(np_query_labels, k=5):
    """
    Given the binary prediction of labels, this function returns the average precision at specified k
    
    Args:
        np_query_labels : numpy array/ python list with binary values
        k   : cutoff point to find the average precision
    
    Returns:
        Average Precision at k
    """
    ap = 0.0
    running_positives = 0
    for idx, i in enumerate(np_query_labels[:k]):
        if i == 1:
            running_positives += 1
            ap_at_count = running_positives/(idx+1)
            ap += ap_at_count
    return ap/k


def get_preds_and_visualize(best_matches, query_gt_dict, img_dir, top_k_to_plot):
    """
    Given the best matching file names and ground truth dictionary for query, this function
    returns the binary prediction as well as plots the top_k image results

    Args:
        best_matches    : list of best file matches. i.e.: ['all_souls_0000051.jpg', .....]
        query_gt_dict   : Dictionary indicating the postive and negative ground truth for query
        img_dir         : Image directory that has all the target images stored
        top_k_to_plot   : number of top results to plot
    
    Returns:
        binary predictions
    """
    # Create python list to store preds
    preds = []

    # For plotting, initialize the figures
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = int(top_k_to_plot/4)
    rows = int(top_k_to_plot/columns)

    for i, pic in enumerate(best_matches): 
        img_name = "{}".format(pic.split("/")[-1])
        color_code = None
        if img_name in query_gt_dict['positive']:
            color_code = (0, 255, 0)
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            color_code = (255, 255, 0)
            preds.append(0)
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
    """
    See gets_preds_and_visualize(**args). 
    It is the same function without any plots
    """
   
    # Create python list to store preds
    preds = []

    # Iterate through the best matches and find predictions
    for i, pic in enumerate(best_matches): 
        img_name = "{}".format(pic.split("/")[-1])
        if img_name in query_gt_dict['positive']:
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            preds.append(0)
        else:
            preds.append(0)
        
    return preds

def get_gt_web(best_matches, query_gt_dict):
    """
    Given best matches and query ground truth, return list of ground truths corresponding to best_matches (for deployment use)

    Args:
        best_matches    : list of best matching files
        query_gt_dict   : dictionary indicating the positive and negative examples for the query

    Returns:
        list of ground truths corresponding to best_matches
    """
    # Create python list to store preds
    preds = []

    # Iterate through the best matches and find predictions
    for i, pic in enumerate(best_matches): 
        img_name = "{}".format(pic.split("/")[-1])
        if img_name in query_gt_dict['positive']:
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            preds.append(-1)
        else:
            preds.append(0)
        
    return preds

def ap_per_query(best_matches, query_gt_dict):
    """
    Given best matches and query ground truth, calculate the average precision

    Args:
        best_matches    : list of best matching files
        query_gt_dict   : dictionary indicating the positive and negative examples for the query

    Returns:
        Average Precision for the query
    """
    # Create python list to store preds
    preds = []

    # Iterate through the best matches and find predictions
    for i, pic in enumerate(best_matches): 
        img_name = "{}".format(pic.split("/")[-1])
        if img_name in query_gt_dict['positive']:
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            preds.append(0)
        else:
            preds.append(0)
    
    num_gt = len(query_gt_dict['positive'])

    return ap_at_k_per_query(preds, k=num_gt)


def plot_history(train_hist, val_hist, y_label, filename, labels=["train", "validation"]):
    """
    Plot training and validation history
    
    Args:
        train_hist: numpy array consisting of train history values (loss/ accuracy metrics)
        valid_hist: numpy array consisting of validation history values (loss/ accuracy metrics)
        y_label: label for y_axis
        filename: filename to store the resulting plot
        labels: legend for the plot
        
    Returns:
        None
    """
    # Plot loss and accuracy
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label = labels[0])
    plt.plot(val_hist, label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()


def center_crop_numpy(img, cropx, cropy):
    """
    Givenn an image numpy array, perform a center crop.

    Args:
        img     : numpy image array
        cropx   : width of crop
        cropy   : height of crop
    
    Returns:
        cropped numpy image array
    """
    y,x = img.shape[:-1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def perform_pca_on_single_vector(ft_np_vector, n_components=2, reshape_dim=2048):
    """
    Given a feature vector, perform dimension reduction using PCA.

    Args:
        ft_np_vector    : numpy feature vector
        n_components    : number of principal components
        reshape_dim     : height of reshaped matrix
    
    Returns
        PCA performed vector
    """
    pca = PCA(n_components=n_components, whiten=True)
    file_fts = ft_np_vector.reshape(reshape_dim, -1)
    pca.fit(file_fts)
    x = pca.transform(file_fts)
    return x.flatten()


def template_matching(target_img_path, compare_img_path_list, img_dir, top_k=500):
    """
    Given a target image path and list of image paths, get the top k structurally similar image paths

    Args:
        target_img_path         : path of reference image
        compare_img_path_list   : paths of images to be compared as a list
        img_dir                 : Image directory
        top_k                   : Number of top similar image paths to return
    
    Returns:
        top_k structurally similar image paths

    Eg run:
        > files = [os.path.join("./data/oxbuild/images", file) for file in os.listdir("./data/oxbuild/images/")]
        > print(files)
        > template_matching("./data/oxbuild/images/all_souls_000051.jpg", files, "./data/oxbuild/images/", 500)
    """
    
    ssim = []
    for other_img_path in tqdm(compare_img_path_list):
        # load the two input images
        image_target = cv2.imread(os.path.join(img_dir, target_img_path))
        image_other = cv2.imread(os.path.join(img_dir, other_img_path))
        
        image_target = center_crop_numpy(image_target, 500, 500)
        image_other = center_crop_numpy(image_other, 500, 500)

        if image_target.shape != image_other.shape:
            continue

        # convert the images to grayscale
        gray_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
        gray_other = cv2.cvtColor(image_other, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        score = compare_ssim(gray_target, gray_other, full=False)
        ssim.append(score)

    indexes = (-np.array(ssim)).argsort()[:top_k]
    final_results = [compare_img_path_list[index] for index in indexes]
    return final_results