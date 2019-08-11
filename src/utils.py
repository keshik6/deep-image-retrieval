import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.measure import compare_ssim
import cv2
from tqdm import tqdm

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
    y,x = img.shape[:-1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def template_matching(target_img_path, compare_img_path_list, img_dir):
    
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

    indexes = (-np.array(ssim)).argsort()[:500]
    final_results = [compare_img_path_list[index] for index in indexes]
    print(final_results)
    return final_results


# files = [os.path.join("./data/oxbuild/images", file) for file in os.listdir("./data/oxbuild/images/")]
# print(files)
# template_matching("./data/oxbuild/images/all_souls_000051.jpg", files)