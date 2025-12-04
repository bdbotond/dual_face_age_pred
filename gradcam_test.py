import os
os.environ["TF_USE_LEGACY_KERAS"]="1"

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import sys
import pandas as pd
import numpy as np
import time
import copy
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.optim as optim
import torchvision
import torch
import torchvision.transforms as T

from torchvision import datasets, io, models, ops, transforms
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





'''PREDICTIONS'''

def predictions(model,model_name,dataloader,data,name):
    start_time=time.time()
    #load model
    model=model
    model.load_state_dict(torch.load(model_name,map_location=device))
    model.to(device)
    model.eval()
    print('model loaded')

    print('starting test on '+str(name)+"!")

    name=str(name)
    test_dataset=dataloader(data)
    test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=128,pin_memory=True,shuffle=False)

    i=1

    relulist=[model.relu,
    model.layer1[0].relu,
    model.layer1[1].relu,
    model.layer1[2].relu,
    model.layer2[0].relu,
    model.layer2[1].relu,
    model.layer2[2].relu,
    model.layer2[3].relu,
    model.layer3[0].relu,
    model.layer3[1].relu,
    model.layer3[2].relu,
    model.layer3[3].relu,
    model.layer3[4].relu,
    model.layer3[5].relu,
    model.layer4[0].relu,
    model.layer4[1].relu,
    model.layer4[2].relu]


    batchnormlist=[model.bn1,
    model.layer1[0].bn1,
    model.layer1[0].bn2,
    model.layer1[0].bn3,
    model.layer1[1].bn1,
    model.layer1[1].bn2,
    model.layer1[1].bn3,
    model.layer1[2].bn1,
    model.layer1[2].bn2,
    model.layer1[2].bn3,
    model.layer2[0].bn1,
    model.layer2[0].bn2,
    model.layer2[0].bn3,
    model.layer2[1].bn1,
    model.layer2[1].bn2,
    model.layer2[1].bn3,
    model.layer2[2].bn1,
    model.layer2[2].bn2,
    model.layer2[2].bn3,
    model.layer2[3].bn1,
    model.layer2[3].bn2,
    model.layer2[3].bn3,
    model.layer3[0].bn1,
    model.layer3[0].bn2,
    model.layer3[0].bn3,
    model.layer3[1].bn1,
    model.layer3[1].bn2,
    model.layer3[1].bn3,
    model.layer3[2].bn1,
    model.layer3[2].bn2,
    model.layer3[2].bn3,
    model.layer3[3].bn1,
    model.layer3[3].bn2,
    model.layer3[3].bn3,
    model.layer3[4].bn1,
    model.layer3[4].bn2,
    model.layer3[4].bn3,
    model.layer3[5].bn1,
    model.layer3[5].bn2,
    model.layer3[5].bn3,
    model.layer4[0].bn1,
    model.layer4[0].bn2,
    model.layer4[0].bn3,
    model.layer4[1].bn1,
    model.layer4[1].bn2,
    model.layer4[1].bn3,
    model.layer4[2].bn1,
    model.layer4[2].bn2,
    model.layer4[2].bn3]

    convlist=[model.conv1,
    model.layer1[0].conv1,
    model.layer1[0].conv2,
    model.layer1[0].conv3,
    model.layer1[1].conv1,
    model.layer1[1].conv2,
    model.layer1[1].conv3,
    model.layer1[2].conv1,
    model.layer1[2].conv2,
    model.layer1[2].conv3,
    model.layer2[0].conv1,
    model.layer2[0].conv2,
    model.layer2[0].conv3,
    model.layer2[1].conv1,
    model.layer2[1].conv2,
    model.layer2[1].conv3,
    model.layer2[2].conv1,
    model.layer2[2].conv2,
    model.layer2[2].conv3,
    model.layer2[3].conv1,
    model.layer2[3].conv2,
    model.layer2[3].conv3,
    model.layer3[0].conv1,
    model.layer3[0].conv2,
    model.layer3[0].conv3,
    model.layer3[1].conv1,
    model.layer3[1].conv2,
    model.layer3[1].conv3,
    model.layer3[2].conv1,
    model.layer3[2].conv2,
    model.layer3[2].conv3,
    model.layer3[3].conv1,
    model.layer3[3].conv2,
    model.layer3[3].conv3,
    model.layer3[4].conv1,
    model.layer3[4].conv2,
    model.layer3[4].conv3,
    model.layer3[5].conv1,
    model.layer3[5].conv2,
    model.layer3[5].conv3,
    model.layer4[0].conv1,
    model.layer4[0].conv2,
    model.layer4[0].conv3,
    model.layer4[1].conv1,
    model.layer4[1].conv2,
    model.layer4[1].conv3,
    model.layer4[2].conv1,
    model.layer4[2].conv2,
    model.layer4[2].conv3]

    ####SET LAYERS###
    target_layers=relulist+batchnormlist+convlist


    for i,(images_test, labels_test) in enumerate(test_dataloader): 
        fpred=[]
        grad_values=[]

        
        with GradCAM(model=model, target_layers=target_layers) as cam:
            batch_size = len(images_test)
            targets = [ClassifierOutputTarget(0) for _ in range(batch_size)]

            grayscale_cam = cam(input_tensor=images_test, targets=targets)

            grad1 = grayscale_cam
            '''model predictions'''
            model_outputs = cam.outputs.cpu().detach().numpy()
            print(model_outputs)   
            grad_values.extend(grad1)


            ###ENABLE TO SET PREDICTIONS TOO###
            #fpred.extend(model_outputs) 

            grad_relu2=pd.DataFrame({'grad_relu_front':grad_values})


            ###OPTIPNALLY SET THE MODEL TO SAVE PREDICTIONS TOO###
            #predictions2=pd.DataFrame({'front_pred':fpred})
            #grad_relu2=pd.concat([predictions2,grad_relu2],axis=1)
            #grad_relu2=grad_relu2.dropna()

            grad_relu2.to_pickle(f"./{str(name)}_{i}_gradcam.pkl")
            if i%10==0:
                print(i)
            i+=1
    print("finished test on !")
    end_time=time.time()
    print("time for testing: "+str(end_time-start_time)+" sec")
    print("finished test on "+str(name)+"!")


    
###CROPPED FACE####
def img_trf_val(code):
    try:
        img=cv2.imread(str(code))
        max_side=max(img.shape[0],img.shape[1])


        transform=A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, fill=(0,0,0)),
        A.Resize(224,224),
        ToTensorV2()
        ])
        front_transformed=transform(image=img)["image"]
        return front_transformed 
        
    except:
        z=torch.zeros(3,224,224)
    return z


###UNCROPPED FACE####
#for un cropeed faces use this function
#import tensorflow as tf

#from retinaface import RetinaFace

""" def img_trf_val(code):
    img=cv2.imread(str(code))

    image = img.copy()

    try:
        data = RetinaFace.detect_faces(image)
        data = resp
        face = data['face_1']
        facial_area = face['facial_area']
        x, y, x2, y2 = facial_area
    
        im_cropeed=img[y:y2,x:x2]
    except:
        print("No image found ",str(code))
        im_cropeed=img

    max_side=max(im_cropeed.shape[0],im_cropeed.shape[1])

    transform=A.Compose([
    A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, fill=(0,0,0)),
    A.Resize(224,224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),

    ])
    front_transformed=transform(image=im_cropeed)["image"]
    return front_transformed 
     """



class imread_front_val(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['front_path'].tolist()]
        self.ages = [a for a in frame['age'].tolist()]

    def __getitem__(self, index):
        img_conv=img_trf_val(str(self.front_path[index]))
        image, target = img_conv, self.ages[index]
        return image, target

    def __len__(self):
        return len(self.frame)

class imread_side_val(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['side_path'].tolist()]
        self.ages = [a for a in frame['age'].tolist()]

    def __getitem__(self, index):
        img_conv=img_trf_val(str(self.front_path[index]))
        image, target = img_conv, self.ages[index]
        return image, target

    def __len__(self):
        return len(self.frame)


class imread_two_val(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path=[f for f in frame['front_path'].tolist()]
        self.side_path = [f for f in frame['side_path'].tolist()]
        self.ages = [a for a in frame['age'].tolist()]

    def __getitem__(self, index):


        input_tensor_front =img_trf_val(str(self.front_path[index]))
        input_tensor_side =img_trf_val(str(self.side_path[index]))
        input_tensor=torch.cat((input_tensor_front,input_tensor_side),dim=2)
        target = self.ages[index]
        return input_tensor, target

    def __len__(self):
        return len(self.frame)


model=models.resnet50(weights=False)
num_features=model.fc.in_features
model.fc=nn.Sequential(nn.Dropout(0.6),
                nn.Linear(num_features,1),
                      )



torch.set_num_threads(10)



front_model_path='./models/dual_model.pt'
side_model_path='./models/dual_model.pt'
two_mode_path='./models/dual_model.pt'


torch.set_num_threads(10)
if  __name__ == "__main__":
    gpu_id = sys.argv[1]
    print(f"Using GPU ID: {gpu_id}")
    dataset_path=sys.argv[2]
    data=pd.read_csv(dataset_path)
    
    direction=sys.argv[3]  #front, side, two
    if direction=='front':
        model_path=front_model_path
        dataloader=imread_front_val
        if 'front_path' not in data.columns:
            print("Error: 'front_path' column not found in the dataset for 'front' direction.")
            sys.exit(1)
    elif direction=='side':
        model_path=side_model_path
        dataloader=imread_side_val
        if 'side_path' not in data.columns:
            print("Error: 'side_path' column not found in the dataset for 'side' direction.")
            sys.exit(1)

    elif direction=='two':
        model_path=two_mode_path
        dataloader=imread_two_val
        if 'front_path' not in data.columns or 'side_path' not in data.columns:
            print("Error: 'front_path' or 'side_path' column not found in the dataset for 'two' direction.")
            sys.exit(1)
    predictions(model,model_path,dataloader,data,direction)

    else:
        print("Invalid direction argument. Use 'front', 'side', or 'two'.")
        sys.exit(1)


