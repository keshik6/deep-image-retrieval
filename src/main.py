from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from model import TripletLoss, TripletNet, Identity, create_embedding_net
from dataset import QueryExtractor, OxfordDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from inference import inference_on_set
from train import train_model


def main(exp_num=1):
    # Define directories
    labels_dir, image_dir = "./data/oxbuild/gt_files/", "./data/oxbuild/images/"

    # Create Query extractor object
    q_train = QueryExtractor(labels_dir, image_dir, subset="train")
    q_valid = QueryExtractor(labels_dir, image_dir, subset="valid")

    # Get query list and query map
    # triplets_train, triplets_valid = q_train.get_triplets(), q_valid.get_triplets() 
    # print(len(triplets_train), len(triplets_valid))

    # Create transformss
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_train = transforms.Compose([transforms.Resize(280),
                                        transforms.RandomResizedCrop(224),                                 
                                        transforms.ColorJitter(brightness=(0.80, 1.20)),
                                        transforms.RandomHorizontalFlip(p = 0.50),
                                        transforms.RandomRotation(15),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean=mean, std=std),
                                        ])

    transforms_valid = transforms.Compose([transforms.Resize(280),
                                            transforms.CenterCrop(224),                                 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            ])

    # Create dataset
    oxford_train = OxfordDataset(labels_dir, image_dir, q_train, transforms=transforms_train)
    oxford_valid = OxfordDataset(labels_dir, image_dir, q_valid, transforms=transforms_valid)

    # Create dataloader
    train_loader = DataLoader(oxford_train, batch_size=2, num_workers=4, shuffle=True)
    valid_loader = DataLoader(oxford_valid, batch_size=2, num_workers=4, shuffle=False)

    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Create embedding network
    embedding_model = create_embedding_net()
    model = TripletNet(embedding_model)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Create log file
    log_file =  open(os.path.join("./results", "log-{}.txt".format(exp_num)), "w+")
    log_file.write("----------Experiment {}----------\n".format(exp_num))

    # Train
    tr_hist, val_hist = train_model(model, device, optimizer, scheduler, train_loader, valid_loader,  
                    save_dir="./weights/", model_name="triplet_model.pth", 
                    epochs=40, log_file=log_file, update_batch=10)

    # Close the file
    log_file.close()

if __name__ == '__main__':
    main()