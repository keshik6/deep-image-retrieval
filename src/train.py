from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from model import TripletLoss, TripletNet, Identity
from dataset import QueryExtractor, OxfordDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from inference import inference_on_set


def train_model(model, device, optimizer, scheduler, train_loader, valid_loader,  
                save_dir="./weights/", model_name="triplet.pth", 
                epochs=20, log_file=None):
    """
    Train a deep neural network model
    
    Args:
        model: pytorch model object
        device: cuda or cpu
        optimizer: pytorch optimizer object
        scheduler: learning rate scheduler object that wraps the optimizer
        train_dataloader: training  images dataloader
        valid_dataloader: validation images dataloader
        save_dir: Location to save model weights, plots and log_file
        epochs: number of training epochs
        log_file: text file instance to record training and validation history
        
    Returns:
        Training history and Validation history (loss and average precision)
    """
    tr_loss, tr_map = [], []
    val_loss, valid_map = [], []
    best_val_map = 0.0
    weights_path = os.path.join(save_dir, model_name)
    infer=False
    
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        
        print("-------Epoch {}----------".format(epoch+1))

        criterion = TripletLoss()

        if (epoch+1)%20 == 0:
            train_loader.dataset.increase_difficulty()
            valid_loader.dataset.increase_difficulty()
            #criterion.reduce_margin()
        
        if epoch != 0:
            infer = True
            scheduler.step()
            

        for phase in ['train', 'valid']:
            running_loss = 0.0
            
            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                for anchor, positive, negative in tqdm(train_loader):
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Retrieve the embeddings
                    output1, output2, output3 = model(anchor, positive, negative)
                    
                    # Calculate the triplet loss
                    loss = criterion(output1, output2, output3)
                    
                    # Sum up batch loss
                    running_loss += loss
                    
                    # Backpropagate the system the determine the gradients
                    loss.backward()
                    
                    # Update the paramteres of the model
                    optimizer.step()

                    # clear variables
                    del anchor, positive, negative, output1, output2, output3
                    gc.collect()
                    torch.cuda.empty_cache()

                if infer:    
                    num_samples = float(len(train_loader.dataset))
                    tr_loss_ = running_loss.item()/num_samples
                    tr_map_ = inference_on_set(subset="train", weights_path=weights_path, top_k=50, device=device)
                    tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                    print('train_loss: {:.4f}\ttrain_mAP: {:.4f}'.format(tr_loss_, tr_map_))
                
                else:
                    num_samples = float(len(train_loader.dataset))
                    tr_loss_ = running_loss.item()/num_samples
                    tr_loss.append(tr_loss_)
                    print('train_loss: {:.4f}'.format(tr_loss_))
            
            else:
                model.train(False)

                with torch.no_grad():
                    for anchor, positive, negative in tqdm(valid_loader):
                        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                        
                        # Create five cropped embeddings
                        bs, ncrops, c, h, w = anchor.size()

                        # Retrieve the embeddings
                        output1, output2, output3 = model(anchor.view(-1, c, h, w), positive.view(-1, c, h, w), negative.view(-1, c, h, w))

                        output1 = output1.view(bs, ncrops, -1).mean(1)
                        output2 = output2.view(bs, ncrops, -1).mean(1)
                        output3 = output3.view(bs, ncrops, -1).mean(1)

                        # Calculate the triplet loss
                        loss = criterion(output1, output2, output3)
                        
                        # Sum up batch loss
                        running_loss += loss
                        
                        # clear variables
                        del anchor, positive, negative, output1, output2, output3
                        gc.collect()
                        torch.cuda.empty_cache()

                if infer:    
                    num_samples = float(len(valid_loader.dataset))
                    valid_loss_ = running_loss.item()/num_samples
                    valid_map_ = inference_on_set(subset="valid", weights_path=weights_path, top_k=50, device=device)
                    valid_loss.append(valid_loss_), valid_map.append(valid_map_)
                    print('valid_loss: {:.4f}\tvalid_mAP: {:.4f}'.format(valid_loss_, valid_map_))

                    if best_val_map < valid_map_:
                        best_val_map = valid_map_
                        print("Saving best weights...")
                        torch.save(model.state_dict(), weights_path=weights_path)
                
                else:
                    num_samples = float(len(valid_loader.dataset))
                    valid_loss_ = running_loss.item()/num_samples
                    tr_loss.append(valid_loss_)
                    print('valid_loss: {:.4f}\t'.format(valid_loss_))
                    torch.save(model.state_dict(), weights_path=weights_path)

                
    return ([tr_loss, tr_map], [val_loss, valid_map])
    


# Define directories
labels_dir, image_dir = "./data/oxbuild/gt_files/", "./data/oxbuild/images/"

# Create Query extractor object
q_train = QueryExtractor(labels_dir, image_dir, subset="train")
q_valid = QueryExtractor(labels_dir, image_dir, subset="valid")

# Get query list and query map
query_names_train, query_map_train = q_train.get_query_names(), q_train.get_query_map()
query_names_valid, query_map_valid = q_valid.get_query_names(), q_valid.get_query_map()

# Create transformss
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transforms_train = transforms.Compose([transforms.Resize(280),
                                      transforms.RandomResizedCrop(256),                                 
                                      transforms.ColorJitter(brightness=(0.80, 1.20)),
                                      transforms.RandomHorizontalFlip(p = 0.50),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=mean, std=std),
                                      ])

transforms_valid = transforms.Compose([transforms.Resize(280),
                                        transforms.FiveCrop(256),                                 
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        ])

# Create dataset
oxford_train = OxfordDataset(labels_dir, image_dir, query_names_train, query_map_train, transforms=transforms_train)
oxford_valid = OxfordDataset(labels_dir, image_dir, query_names_valid, query_map_valid, transforms=transforms_valid)

# Create dataloader
train_loader = DataLoader(oxford_train, batch_size=4, num_workers=0, shuffle=True)
valid_loader = DataLoader(oxford_valid, batch_size=4, num_workers=0, shuffle=False)

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
model.to(device)

# Create optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=7.5e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Train
train_model(model, device, optimizer, scheduler, train_loader, valid_loader,  
                save_dir="./weights/", model_name="triplet_model.pth", 
                epochs=50, log_file=None)

