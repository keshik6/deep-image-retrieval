from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from model import TripletNet, create_embedding_net
from dataset import QueryExtractor, VggImageRetrievalDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from train import train_model
from utils import plot_history
import math

def main(data_dir, results_dir, weights_dir,
        which_dataset, image_resize, image_crop_size, 
        exp_num,
        max_epochs, batch_size, samples_update_size, 
        num_workers=4, lr=5e-6, weight_decay=1e-5):
    """
    This is the main function. You need to interface only with this function to train. (It will record all the results)
    Once you have trained use create_db.py to create the embeddings and then use the inference_on_single_image.py to test
    
    Arguments:
        data_dir    : parent directory for data
        results_dir : directory to store the results (Make sure you create this directory first)
        weights_dir : directory to store the weights (Make sure you create this directory first)
        which_dataset : "oxford" or "paris" 
        image_resize : resize to this size
        image_crop_size : square crop size
        exp_num     : experiment number to record the log and results
        max_epochs  : maximum epochs to run
        batch_size  : batch size (I used 5)
        samples_update_size : Number of samples the network should see before it performs one parameter update (I used 64)
    
    Keyword Arguments:
        num_workers : default 4
        lr      : Initial learning rate (default 5e-6)
        weight_decay: default 1e-5

    Eg run:
        if __name__ == '__main__':
            main(data_dir="./data/", results_dir="./results", weights_dir="./weights",
            which_dataset="oxbuild", image_resize=460, image_crop_size=448,
            exp_num=3, max_epochs=10, batch_size=5, samples_update_size=64)
    """
    # Define directories
    labels_dir = os.path.join(data_dir, which_dataset, "gt_files")
    image_dir = os.path.join(data_dir, which_dataset, "images")

    # Create Query extractor object
    q_train = QueryExtractor(labels_dir, image_dir, subset="train")
    q_valid = QueryExtractor(labels_dir, image_dir, subset="valid")

    # Create transformss
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_train = transforms.Compose([transforms.Resize(image_resize),
                                        transforms.RandomResizedCrop(image_crop_size, scale=(0.8, 1.2)),                                 
                                        transforms.ColorJitter(brightness=(0.80, 1.20)),
                                        transforms.RandomHorizontalFlip(p = 0.50),
                                        transforms.RandomChoice([
                                            transforms.RandomRotation(15),
                                            transforms.Grayscale(num_output_channels=3),
                                        ]),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean=mean, std=std),
                                        ])

    transforms_valid = transforms.Compose([transforms.Resize(image_resize),
                                            transforms.CenterCrop(image_crop_size),                                 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            ])

    # Create dataset
    dataset_train = VggImageRetrievalDataset(labels_dir, image_dir, q_train, transforms=transforms_train)
    dataset_valid = VggImageRetrievalDataset(labels_dir, image_dir, q_valid, transforms=transforms_valid)

    # Create dataloader
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2020)
    torch.manual_seed(2020)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create embedding network
    embedding_model = create_embedding_net()
    model = TripletNet(embedding_model)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Create log file
    log_file =  open(os.path.join(results_dir, "log-{}.txt".format(exp_num)), "w+")
    log_file.write("----------Experiment {}----------\n".format(exp_num))
    log_file.write("Dataset = {}, Image sizes = {}, {}\n".format(which_dataset, image_resize, image_crop_size))

    # Creat batch update value
    update_batch = int(math.ceil(float(samples_update_size)/batch_size))
    model_name = "{}-exp-{}.pth".format(which_dataset, exp_num)
    loss_plot_save_path = os.path.join(results_dir, "{}-loss-exp-{}.png".format(which_dataset, exp_num))

    # Print stats before starting training
    print("Running VGG Image Retrieval Training script")
    print("Dataset used\t\t:{}".format(which_dataset))
    print("Max epochs\t\t: {}".format(max_epochs))
    print("Gradient update\t\t: every {} batches ({} samples)".format(update_batch, samples_update_size))
    print("Initial Learning rate\t: {}".format(lr))
    print("Image resize, crop size\t: {}, {}".format(image_resize, image_crop_size))
    print("Available device \t:", device)

    # Train the triplet network
    tr_hist, val_hist = train_model(model, device, optimizer, scheduler, train_loader, valid_loader,  
                    epochs=max_epochs, update_batch=update_batch, model_name=model_name, 
                    save_dir=weights_dir, log_file=log_file)

    # Close the file
    log_file.close()

    # Plot and save
    plot_history(tr_hist, val_hist, "Triplet Loss", loss_plot_save_path, labels=["train", "validation"])

    
# if __name__ == '__main__':
#     main(data_dir="./data/", results_dir="./results", weights_dir="./weights",
#         which_dataset="oxbuild", image_resize=460, image_crop_size=448,
#         exp_num=3, max_epochs=10, batch_size=5, samples_update_size=64)