import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from retinaface import RetinaFace
import sys


def img_transform_dual(image_code_f,image_code_s):
    try:
        image_front=cv2.imread(str(image_code_f))
    except:
        raise ValueError(f"Front image at {image_code_f} path not found, or not a valid image file")
    try:
        faces = RetinaFace.extract_faces(img_path = (str(image_code_f)), align = True)
        image_front=faces[0]
    except:
        print("Retainface cannot find the face on the image, using the full image for the prediction")
        # If no detected face return the original image
    max_side=max(image_front.shape[0],image_front.shape[1])
    transform=A.Compose([
    A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, value=(0,0,0)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.Resize(224,224),
    ToTensorV2()
    ])

    front_transformed=transform(image=image_front)["image"]

    try:
        image_side=cv2.imread(str(image_code_s))
    except:
        raise ValueError(f"Side image at {image_code_s} path not found, or not a valid image file")
    try:
        faces = RetinaFace.extract_faces(img_path = (str(image_code_s)), align = True)
        image_side=faces[0]
    except:
        print("Retainface cannot find the face on the image, using the full image for the prediction")
        # If no detected face return the original image
    max_side=max(image_side.shape[0],image_side.shape[1])
    transform=A.Compose([
    A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, value=(0,0,0)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.Resize(224,224),
    ToTensorV2()
    ])




    side_transformed=transform(image=image_side)["image"]
    full_img=torch.cat((front_transformed,side_transformed),axis=1)

    return full_img 



def img_trf(code):
    try:
        img=cv2.imread(str(code))
    except:
        raise ValueError(f"Image at {code} path not found, or not a valid image file")
    try:
        faces = RetinaFace.extract_faces(img_path = (str(code)), align = True)
        img=faces[0]
    except:
        print("Retainface cannot find the face on the image, using the full image for the prediction")
        # If no detected face return the original image
        img=img
    max_side=max(img.shape[0],img.shape[1])
    #print(max_side)
    #add transform
    transform=A.Compose([
    A.PadIfNeeded(p=1.0, min_height=max_side, min_width=max_side, border_mode=0, value=(0,0,0)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.Resize(224,224),
    ToTensorV2()
    ])
    front_transformed=transform(image=img)["image"]
    return front_transformed 

model=models.resnet50(weights=None)
num_features=model.fc.in_features
model.fc=nn.Sequential(nn.Dropout(0.6),
    nn.Linear(num_features, 1))
    
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class imread_front_val(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['front_path'].tolist()]

    def __getitem__(self, index):
        img_conv=img_trf(str(self.front_path[index]))
        return img_conv

    def __len__(self):
        return len(self.frame)

class imread_side_val(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['side_path'].tolist()]

    def __getitem__(self, index):
        img_conv=img_trf(str(self.front_path[index]))
        return img_conv

    def __len__(self):
        return len(self.frame)
    
class imread_dual(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame=frame
        self.front_path = [f for f in frame['front_path'].tolist()]
        self.side_path = [f for f in frame['side_path'].tolist()]

    def __getitem__(self, index):
        img_conv=img_transform_dual(str(self.front_path[index]),str(self.side_path[index]))
        return img_conv

    def __len__(self):
        return len(self.frame)

def predictions(direction,data):
    if str(direction)=='front':
        model.load_state_dict(torch.load('./models/front_model.pt'))
        dataloader=imread_front_val
    elif str(direction)=='side':
        model.load_state_dict(torch.load('./models/side_model.pt'))
        dataloader=imread_side_val
    elif str(direction)=='dual':
        model.load_state_dict(torch.load('./models/dual_model.pt'))
        dataloader=imread_dual
    else:
        raise ValueError("Invalid direction specified. Use 'front', 'side', or 'dual'.")
    model.to(device)
    model.eval()
    print('Model loaded')


    name=str(data)
    test_dataset=dataloader(data)
    test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=64,pin_memory=False,shuffle=False,num_workers=1)

    predictions_front=[]
    i=1
    for images_test in test_dataloader: 
        images_test=images_test.to(device=device, dtype=torch.float)
        
        pred_front = model(images_test) 
        
        predict_front,_=torch.max(pred_front.data,1) 
        predictions_front.append(predict_front.cpu().numpy())

        if len(predictions_front)%1000==0:
            print(len(predictions_front))
    test_data_full=data.reset_index(drop=True)
    f_pred=list(np.concatenate(predictions_front, axis=0 ))
    predictions=pd.DataFrame({'predictions':f_pred})
    test_with_pred=pd.concat([test_data_full,predictions],axis=1)
    test_with_pred=test_with_pred.dropna()
    print(len(test_with_pred))

    test_with_pred.to_csv("./predictions.csv",index=False)
    print("finished test on "+str(name)+"!")



#pred=predict(device,'front','./test.jpg')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <direction> <csv_path>")
        sys.exit(1)
    
    direction = sys.argv[1]
    csv_path = sys.argv[2]
    data = pd.read_csv(csv_path).head(10)
    prediction = predictions(direction, data)


