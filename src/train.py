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
                epochs=20, log_file=None, db=["./data/oxbuild/images/", "./fts"]):
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
    valid_loss, valid_map = [], []
    best_val_map = 0.0
    weights_path = os.path.join(save_dir, model_name)
    temp_weights_path = os.path.join(save_dir, "temp-{}".format(model_name))
    infer=False
    
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("-------Epoch {}----------".format(epoch+1))

        criterion = TripletLoss(margin=3)

        if (epoch+1)%20 == 0:
            train_loader.dataset.increase_difficulty()
            valid_loader.dataset.increase_difficulty()
            criterion.reduce_margin()
        
        if epoch != 0:
            print("> Modifying learning rate")
            scheduler.step()
            
        for phase in ['train', 'valid']:
            running_loss = 0.0
            
            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                print("> Training the network")
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


                # Save trained model
                print("> Saving trained weights...")
                torch.save(model.state_dict(), temp_weights_path)
                

                # Calculate statistics and log
                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_map_, valid_map_ = inference_on_set(model=model, top_k=50, device=device)
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                print('> train_loss: {:.4f}\ttrain_mAP: {:.4f}'.format(tr_loss_, tr_map_))
                log_file.write('> train_loss: {:.4f}\ttrain_mAP: {:.4f}\n'.format(tr_loss_, tr_map_))
                
            
            else:
                model.train(False)

                print("> Running validation on the network")
                with torch.no_grad():
                    for anchor, positive, negative in tqdm(valid_loader):
                        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                        
                        # Retrieve the embeddings
                        output1, output2, output3 = model(anchor, positive, negative)
                        
                        # Calculate the triplet loss
                        loss = criterion(output1, output2, output3)
                        
                        # Sum up batch loss
                        running_loss += loss
                        
                        # clear variables
                        del anchor, positive, negative, output1, output2, output3
                        gc.collect()
                        torch.cuda.empty_cache()

                # Get statistics and log
                num_samples = float(len(valid_loader.dataset))
                valid_loss_ = running_loss.item()/num_samples
                valid_loss.append(valid_loss_), valid_map.append(valid_map_)
                print('> valid_loss: {:.4f}\tvalid_mAP: {:.4f}'.format(valid_loss_, valid_map_))
                log_file.write('> valid_loss: {:.4f}\tvalid_mAP: {:.4f}\n'.format(valid_loss_, valid_map_))
                
                # If improvement in mAP is observed, best weights = temporary weights
                if best_val_map < valid_map_:
                    best_val_map = valid_map_
                    print("> Saving best weights...")
                    log_file.write("Saving best weights...\n")
                    torch.save(model.state_dict(), weights_path)

                
    return ([tr_loss, tr_map], [valid_loss, valid_map])
    


