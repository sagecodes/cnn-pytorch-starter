import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import matplotlib.pyplot as plt                        
from tqdm import tqdm


def train(model, n_epochs, loaders, optimizer,
                    criterion, device, save_path, verbose=False):
    '''
    This function trains a pytorch model and saves model when loss decreases

    Example use:
        train(res50.model,
                10,
                { 'train':train_loader, 'valid':val_loader },
                optim.SGD(res50.model.fc.parameters(), lr=learn_rate),
                nn.CrossEntropyLoss(),
                'cuda',
                trained_models/test_train.pt)

    args:
        model (pytorch_model): the model you would like to train 
        n_epochs (number): the number of epochs to train model 
        loaders (Dict): Dictionary of pytorch data loaders.
                        Expects 'train' and 'valid'
        optimizer (pytorch model optimizer): optimizer for training
        criterion (pytorch model criterion): criterion for training
        device (string): decide to run training on. usually 'cpu' or 'gpu'
        save_path (string): the path where you would like to save the model and
            training history. only inlucde path & training name. No extentions

    output:
        saves model to 'save_path' with extension .pt
        saves training history to 'save_path' with extension .csv
    returns: training history list
    '''
    # history dict for training log    
    history = {'train_loss':[],
                'train_acc':[],
                'val_loss':[],
                'val_acc':[]
                }

    model = model.to(device)
    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 

    with tqdm(range(n_epochs)) as t:
        for epoch in range(1, n_epochs+1):
            
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            num_correct = 0       
            num_examples = 0
        
            ######################  
            # train the model    #
            ######################  
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

                # calculate training accuracy
                train_acc = num_correct / num_examples

                # display train loss while training
                t.set_postfix(loss='{:.3f}'.format(train_loss))
                

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
                
                # caluculate validation accuracy
                correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
                                            target).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
                valid_acc = num_correct / num_examples
                
                # Display val accuracy while training
                t.set_postfix(val_accuracy='{:.3f}'.format(valid_acc))

            # print training data per epoch
            print("\n---------------------------------\n")
            print(f'epoch: {epoch}/{n_epochs}')
            print(f'train_loss:__{train_loss:.6f}')
            print(f'train_acc:___{train_acc:.6f}')
            print(f'val_loss:____{valid_loss:.6f}')
            print(f'val_acc:_____{valid_acc:.6f}')

            ## save the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                torch.save(model.state_dict(), save_path+'.pt')
                print(f'\nval_loss decreased: model saved at {save_path}.pt')
                valid_loss_min = valid_loss

            print("\n---------------------------------\n")

            # append training & validation history
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(float(valid_loss))
            history['val_acc'].append(valid_acc)

            # Save training history to csv file
            save_history_csv(history, save_path)
            
            # update epoch progress bar
            t.update()
    
    return history

def plot_train_history(history, num_epochs):
    """
    this function plots the training history as a line graph

    example:
        plot_train_history(history, 32)

    
    args:
        history (dict) : dictionary containing lists for train/val loss & accuracy
                        expects the following key value pairs:
                                train_loss (list), val_loss(list),
                                train_acc(list), val_acc(list)

        num_epochs (num): the number of epochs the model was trained on


    TODO: 
        - Get number of epochs from list length
    
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, num_epochs),
                history["train_loss"],
                label="train_loss")
    plt.plot(np.arange(0, num_epochs),
                history["val_loss"],
                label="val_loss")
    plt.plot(np.arange(0, num_epochs),
                history["train_acc"],
                label="train_acc")
    plt.plot(np.arange(0, num_epochs),
                history["val_acc"],
                label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    plt.close()


def save_history_csv(history, save_path):
    """
    this function saves model training history as a csv file

    example:
        save_history_csv(history, 'train_hist')

    
    args:
        history (dict) : dictionary containing lists for train/val loss & accuracy
                        expects the following key value pairs:
                                train_loss (list), val_loss(list),
                                train_acc(list), val_acc(list)

        save_path (string): the path where you would like to save file.
        Including file name, but not extension.


    output:
        saves training history with name passed in 'save_path'
        prints 'Saving file at {save_path}'
    
    """

    df = pd.DataFrame(
    {'train_loss': history["train_loss"],
     'val_loss': history["val_loss"],
     'train_acc': history["train_acc"],
     'val_acc': history["val_acc"]
    })

    print(f"Saving history at {save_path}.csv\n")

    df.to_csv(save_path+'.csv')


def predict(model, image, device, path=False):
    """
    this function returns a class prediction for an image from a model

    example:
        predict(res50.model, "animals/cata/cat1.jpg", "cuda):

    
    args:
        model (pytorch_model): the model you would like to train 
        device (string): decide to run training on. usually 'cpu' or 'gpu'
        img_path (string): the path to image for prediction.
                            Including file name & extension.

    returns:
        predicted index(int) for image
    
    TODO:
        Verbose option for top preditions while predicting
    """
    if path == True:
        image = Image.open(image)
    
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
      
    # return only highest prediction index
    return prediction



def save_model(model, save_path):
        '''
        This function saves a pytorch model using state_dict() method

        Example use:
            save_model(res_model , 'test_save_method2.pt')

        args:
            model (pytorch_model): the model you would like to save
            save_path (string): the path where you would like to save file.
              Including file name & extension.

        output:
            saves model with name passed in 'save_path'
            prints 'Model saved at: {save_path}'

        TODO:
            Check if model name already exsists & overwrite options`

        '''

        torch.save(model.state_dict(), save_path)
        print(f'Model saved at: {save_path}')


def load_model(model, load_path, evals=False):
        '''
        This function loads a pytorch model using load_state_dict() method 

        Example use:
            load_model(resnet.model , 'test_train.pt', True)

        args:
            model (pytorch_model): the model architecture you would like to load
                weights into 
            load_path (string): the path to the file you want to load. 
                Including file name & extension.
            evals (Boolean): decision for loading weights ready for prediction
                or more trainin. True = prediction ready.

        output:
            returns model with weights passed in from 'load_path' file
        '''

        model.load_state_dict(torch.load(load_path))
        
        if evals:
            model.eval()

        return model
