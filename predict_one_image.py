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
    A.ToTensorV2()
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
    A.ToTensorV2()
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


def predict(direction,image_path,side_image_path):
    if str(direction)=='front':
        model.load_state_dict(torch.load('./models/front_model.pt'))
        img_trf_mode=img_trf
        image=img_trf_mode(image_path)

    elif str(direction)=='side':
        model.load_state_dict(torch.load('./models/side_model.pt'))
        img_trf_mode=img_trf
        image=img_trf_mode(image_path)

    elif str(direction)=='dual':
        model.load_state_dict(torch.load('./models/dual_model.pt'))
        img_trf_mode=img_transform_dual
        image=img_transform_dual(image_path, side_image_path)
    else:
        raise ValueError("Invalid direction specified. Use 'front', 'side', or 'dual'.")
    print('Model loaded')
    model.to(device)
    model.eval()
    prediction=model(image.unsqueeze(0).to(device))
    prediction=prediction.cpu().detach().numpy()
    return prediction



#pred=predict(device,'front','./test.jpg')

if __name__ == "__main__":
    if sys.argv[1] =='dual':
        if len(sys.argv) != 4:
            print("Usage: python predict.py <direction> <image_front_path> <image_side_path>")
            sys.exit(1)
        direction = sys.argv[1]
        image_front_path = sys.argv[2]
        image_side_path = sys.argv[3]
        prediction = predict(direction, image_front_path, image_side_path)
        print(f"Prediction: {prediction}")

    elif len(sys.argv) == 3:
        #sys.exit(1)
    
        direction = sys.argv[1]
        image_path = sys.argv[2]
        
        prediction = predict(direction, image_path,None)
        print(f"Prediction: {prediction}")
    else:
        print("Usage: python predict.py <direction> <image_path>")
        sys.exit(1)

