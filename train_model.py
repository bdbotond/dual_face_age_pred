import sys
import os 
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 1:
    try:
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
        print("TensorFlow is configured to use the second GPU.")
    except:
        # Invalid device or other error
        pass

import json

import numpy as np
import pandas as pd
import polars as pl
import time
import copy
from retinaface import RetinaFace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split,GroupShuffleSplit 
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim.lr_scheduler as lr_scheduler


torch.set_num_threads(10)
torch.backends.cudnn.benchmark = True


#IMAGETRANSFORM

def img_trf_train(code):
    try:
        original_img = cv2.cvtColor(cv2.imread(str(code)), cv2.COLOR_BGR2RGB)
        try:
            faces = RetinaFace.extract_faces(img_path = (str(code)), align = True)
            img=faces[0]
            #print(img.shape)
        except:
            img=original_img
            print('no face found')
        max_side=max(img.shape[0],img.shape[1])
        #print(max_side)
        #add transform
        transform=A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.4,1.1), shear=(-10,10), rotate=(-30,30), p=0.8,fit_output=True,keep_ratio=True,border_mode=0,fill=0),
        A.GridDistortion (num_steps=5, distort_limit=(-0.3, 0.3), interpolation=1, p=0.1),
        A.RandomBrightnessContrast(p=0.5),
        A.ChannelShuffle(p=0.2),
        A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, fill=(0,0,0)),
        A.Resize(224,224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])
        front_transformed=transform(image=img)["image"]
        return front_transformed 
    except:
        print('error')
        z=torch.zeros(3,224,224)
    return z 


    
def img_trf_val(code):
    try:
        original_img = cv2.cvtColor(cv2.imread(str(code)), cv2.COLOR_BGR2RGB)
        try:
            faces = RetinaFace.extract_faces(img_path = (str(code)), align = True)
            img=faces[0]
        except:
            print('no face found')
            img=original_img
        max_side=max(img.shape[0],img.shape[1])
        #print(max_side)
        #add transform
        transform=A.Compose([
        A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, fill=(0,0,0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Resize(224,224),
        ToTensorV2()
        ])
        front_transformed=transform(image=img)["image"]
        return front_transformed 
        
    except:
        z=torch.zeros(3,224,224)
        print('error')
    return z   


class dataloader_train(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['path'].tolist()]
        self.ages = [a for a in frame['age'].tolist()]

    def __getitem__(self, index):
        img_conv=img_trf_train(str(self.front_path[index]))
        image, target = img_conv, self.ages[index]
        return image, target

    def __len__(self):
        return len(self.frame)
    

class dataloader_test(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['path'].tolist()]
        self.ages = [a for a in frame['age'].tolist()]

    def __getitem__(self, index):
        img_conv=img_trf_val(str(self.front_path[index]))
        image, target = img_conv, self.ages[index]
        return image, target

    def __len__(self):
        return len(self.frame)
    

# EARLY STOPPING
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print('Error count: ', self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        


#TRAIN MODEL
def model_train(model,loss_function,optimizer,number_of_epochs,lr,train_dataloader,val_dataloader,scheduler,dataframe):



    train_df,val_df= train_test_split(dataframe,test_size=0.2,random_state=42)
    val_df,test_df= train_test_split(val_df,test_size=0.5,random_state=42)  



    ###test_set is remaining for external testing###


    train_data=train_dataloader(train_df)
    val_data=val_dataloader(val_df)
    train_set=torch.utils.data.DataLoader(train_data, batch_size=200,pin_memory=True,shuffle=True)
    val_set=torch.utils.data.DataLoader(val_data, batch_size=200,pin_memory=True,shuffle=False)




    early_stopping = EarlyStopping(patience=30, min_delta=0.1)


    training_params_list = {
        "loss_function": str(loss_function),
        "optimizer": str(optimizer),
        "number_of_epochs": number_of_epochs,
        "lr": lr,
        "early_stopping_patience": early_stopping.patience,
        "early_stopping_min_delta": early_stopping.min_delta,
        "filter": filter,
        "train_set_batch_size": train_set.batch_size,
        "val_set_batch_size": val_set.batch_size,
        "train_set_size": len(train_set.dataset),
        "val_set_size": len(val_set.dataset),
    }

    print(training_params_list)

    print('start')
    since = time.time()
    best_model=copy.deepcopy(model.state_dict())
    best_loss=100
    best_epoch=0
    best_loss=100
    train_loss=[]
    validation_loss=[]
    mae_list=[]
    lr_list=[]
    #train
    for epoch in range(number_of_epochs):
        model.train()
        loss=0
        for images,labels in train_set:
            #forward 
            images=images.to(device=device, dtype=torch.float) 
            labels=labels.to(device=device, dtype=torch.float) 
            #print(images[1].mean())
            #print(labels)
            #makeing prediction

            output=model(images)
            #print(output.shape)
            #print(labels.shape)
            loss=loss_function(output,labels.unsqueeze(1))
            #backward
            optimizer.zero_grad()
            '''loss.backward()'''
            loss.backward()
            #update weight
            optimizer.step()

        #eval
        model.eval()
        loss_val=0
        with torch.no_grad(): 
            for images_val, labels_val in val_set: 
                images_val=images_val.to(device=device, dtype=torch.float) 
                labels_val=labels_val.to(device=device, dtype=torch.float) 
                output_val = model(images_val) 
                loss_val = loss_function(output_val,labels_val.unsqueeze(1))
                optimizer.zero_grad()#
                  
                predicted_val,_ = torch.max(output_val.data, 1) 
            error_avg=torch.sum(abs(predicted_val-labels_val))/labels_val.size(0)#MAE
            mae_list.append(error_avg.item())
        #print(optimizer.param_groups[0]["lr"])
        scheduler.step(loss_val)
        lrrate=optimizer.param_groups[0]["lr"]
        lr_list.append(lrrate)


        train_loss.append(loss.item())
        
        validation_loss.append(loss_val.item())

        print('Epoch [{}/{}],\nLoss:{:.4f},Validation Loss:{:.4f},MAE:{:.4f},Lr:{}'.format(epoch+1, number_of_epochs, loss.item(), loss_val.item(),error_avg,lrrate))
        print('-'*10)


        if loss_val.item()<=best_loss:
            best_loss=loss_val.item()
            best_epoch=epoch
            print(best_loss,best_epoch)
            best_model=copy.deepcopy(model.state_dict())
            torch.save(best_model,f'./model_data/resnet50_best_loss_delta_.pt')
            


        early_stopping(loss_val.item())
        if early_stopping.early_stop:
            print("We are early stopped at epoch:", epoch)
            break
            
        
            


    time_elapsed=time.time()-since
    print(f'Time elapsed:{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f} seconds')
    #learn_data_front=1
    #load the best model

    model_name=f'resnet50'
    model.load_state_dict(best_model)
    torch.save(best_model,f'./model_data/{model_name}_'+str(round(best_loss,4))+'.pt')
    pd.DataFrame({'train_loss':train_loss,'val_loss':validation_loss,'mae':mae_list,'lr':lr_list}).to_csv(f"./model_data/{model_name}_train_val_loss_history_{round(best_loss,4)}.csv",index=False)
    with open(f'./model_data/{model_name}_train_data_{round(best_loss,4)}.json', 'w') as f:
        json.dump(training_params_list, f, indent=4)
    os.remove(f'./model_data/resnet50_best_loss_delta_.pt')

    return model




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#MODEL BETOLTESE

model=models.resnet50(weights=None)
num_features=model.fc.in_features
model.fc=nn.Sequential(nn.Dropout(0.4),
    nn.Linear(num_features, 1))

for param in model.parameters():
    param.requies_grad=True
for param in model.fc.parameters():
    param.requies_grad=True





model.to(device)


#PARAMETERS

number_of_epochs=200
lr=5e-4


loss_function=nn.MSELoss()
#rmse loss
optimizer=optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)


""" scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                 base_lr = 1e-8, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                 max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
                 step_size_up = 5, # Number of training iterations in the increasing half of a cycle
                 mode = "exp_range") """

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.7, patience=10, verbose=True)



data = pd.read_csv(str(sys.argv[2])).head(10)


#TRAIN
#model_train(model,loss_function,optimizer,number_of_epochs,lr,train_dataloader,val_dataloader,scheduler)
print('finish')


if  __name__ == "__main__":
    gpu_id = sys.argv[1]
    print(f"Using GPU ID: {gpu_id}")

    dataframe_path=sys.argv[2]
    print(f"Using dataframe: {dataframe_path}")
    model_train(model,loss_function,optimizer,number_of_epochs,lr,dataloader_train,dataloader_test,scheduler,data)
