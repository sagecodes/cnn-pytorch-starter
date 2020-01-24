import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

from IPython.display import display, clear_output


def train(model, n_epochs, loaders, optimizer,
                    criterion, device, save_path, verbose=False):

    """returns trained model"""    
    train_output = []

    model = model.to(device)
    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        num_correct = 0       
        num_examples = 0
    
        # train the model
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to device
            data = data.to(device)
            target = target.to(device)

            # reset gradient weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # Compute Gradient
            loss.backward()
            
            # Adjust weights w/ Gradient
            optimizer.step()
            
            ## find the loss and update the model parameters accordingly
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))


            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
                                        target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

            # Output info in jupyter notebook
            
            if verbose:
                print('Epoch #{}, Batch #{} train_loss: {:.6f}'.format(epoch, batch_idx + 1, train_loss))
            else:
                clear_output(wait=True)
                display('Epoch #{}, Batch #{} train_loss: {:.6f} train_acc{:.6}'.format(epoch, batch_idx + 1, train_loss, num_correct / num_examples))
            

        ######################    
        # validate the model #
        ######################
        num_correct = 0       
        num_examples = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            
            # move to device
            data = data.to(device)
            target = target.to(device)
                
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
                                        target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        # append training/validation output to output list 
        train_output.append('Epoch: {} train_loss: {:.6f} val_loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print(('SAVE MODEL: val_loss decrease ({:.6f})'.format(valid_loss)))
            valid_loss_min = valid_loss
    
    self.history = train_output

    # model.log(history)
    # model.load()
    # return trained model
    return model


def predict(model, img_path, device, verbose=False):
    # load the image and return the predicted breed
    
    image = Image.open(img_path)
    
    # Transform set to 244px recommended from pytorch doc 
    # for this pre trained network & change to tensor
    transform = transforms.Compose([transforms.Resize(size=(244, 244)),
                                    transforms.ToTensor()])
                                    
    img = transform(image)[:3,:,:].unsqueeze(0)
    
    # Change to device
    img = img.to(device)
    model = model.to(device)
    
    preds = model(img)
    
    prediction = torch.max(preds,1)[1].item()
    
    print(prediction)
    # if verbose:
    #     print("Predicted class is: {}(index: {})".format(
    #                                             self.class_names[prediction],
    #                                             prediction))        
    # # return only highest prediction index
    return prediction
