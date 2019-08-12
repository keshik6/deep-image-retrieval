from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from model import TripletLoss, TripletNet
import math


def train_model(model, device, optimizer, scheduler, 
                train_loader, valid_loader, 
                epochs, update_batch, model_name,
                save_dir, 
                log_file):
    """
    Train a deep neural network model
    
    Args:
        model   : pytorch model object
        device  : cuda or cpu
        optimizer   : pytorch optimizer object
        scheduler   : learning rate scheduler object that wraps the optimizer
        train_dataloader    : training  images dataloader
        valid_dataloader    : validation images dataloader
        epochs  : number of training epochs
        update_batch    : after how many batches should gradient update be performed 
        save_dir    : Location to save model weights, plots and log_file
        log_file    : text file instance to record training and validation history
        
    Returns:
        Training history and Validation history (loss)
    """
    tr_loss = []
    valid_loss = []
    best_val_loss = np.iinfo(np.int32).max
    weights_path = os.path.join(save_dir, model_name)
    temp_weights_path = os.path.join(save_dir, "temp_{}".format(model_name))
    last_batch = math.ceil(len(train_loader.dataset)/update_batch)
    
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("-------Epoch {}----------".format(epoch+1))

        criterion = TripletLoss(margin=2)
        train_loader.dataset.reset()
        
        if (epoch+1)%update_batch == 0:
            print("> Modifying learning rate")
            scheduler.step()
            
        for phase in ['train', 'valid']:
            running_loss = 0.0
            
            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                print("> Training the network")
                for batch_idx, [anchor, positive, negative] in enumerate(tqdm(train_loader)):
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    
                    # Retrieve the embeddings
                    output1, output2, output3 = model(anchor, positive, negative)
                    
                    # Calculate the triplet loss
                    loss = criterion(output1, output2, output3)
                    
                    # Sum up batch loss
                    running_loss += loss
                    
                    # Backpropagate the system the determine the gradients
                    loss.backward()


                    if (batch_idx + 1)%update_batch == 0 or (batch_idx+1) == last_batch:
                        # Update the paramteres of the model
                        optimizer.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()
                    
                    
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
                tr_loss.append(tr_loss_)
                print('> train_loss: {:.4f}\t'.format(tr_loss_))
                log_file.write('> train_loss: {:.4f}\t'.format(tr_loss_))
                
            
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
                valid_loss.append(valid_loss_)
                print('> valid_loss: {:.4f}\t'.format(valid_loss_))
                log_file.write('> valid_loss: {:.4f}\t'.format(valid_loss_))
                
                
                if valid_loss_ < best_val_loss:
                    best_val_loss = valid_loss_
                    print("> Saving best weights...")
                    log_file.write("Saving best weights...\n")
                    torch.save(model.state_dict(), weights_path)

                
    return (tr_loss, valid_loss)
    


