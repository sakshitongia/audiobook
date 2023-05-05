import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# #from engine import train_one_epoch, evaluate
# #import utils
# #import transforms as T
from PIL import Image
import pickle
import tensorflow as tf
import keras
import io

from keras import models


train = pd.read_csv(r"C:\Users\saksh\data298\application\train_annotations.csv")
classes = train[['class']].value_counts()

labels = ['Cardiac', 'Spiderman', 'Dr Doom', 'Wolverine', 'Cyclops', 'Storm',
       'Jubilee', 'Robot', 'ProfX', 'Captain America', 'PeterParker',
       'Mary Jane', 'Man', 'aunt may', 'Woman', 'Black Cat', 'Boy',
       'Beast', 'Gambit', 'Goblin', 'Mysterio', 'Hulk', 'Rogue',
       'Juggernaut', 'Jean', 'Doc ock', 'Sandman', 'Venom', 'Toad',
       'ScarletWitch', 'QuickSilver', 'Magneto', 'Ironman', 'Brock',
       'Scorpion', 'Apocalypse', 'Electro', 'Vulture', 'Chance',
       'Chameleon', 'Sabertooth', 'Morbius', 'Kingpin', 'Havok', 'OldMan',
       'Leader', 'Carnage', 'Girl', 'Thor', 'Dr Strange', 'Sinister',
       'Boomerang', 'j jonah jameson', 'Rhino']




class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

################### GET MODEL ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

# WEIGHTS_FILE = "C:/Users/saksh/data298/application/demo_model1.pt"

# num_classes = 55

# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features

# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# # Load the traines weights
# model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))
#WEIGHTS_FILE = "C:/Users/saksh/data298/application/demo_model1.pt"
# model = model.to(device)


 ##################### FUNCTION ###############################
def obj_detector(img):

    
    
    #WEIGHTS_FILE = "C:/Users/saksh/data298/application/demo_model1.pt"
    with open("model.pkl", "rb") as f:
        buffer = io.BytesIO(f.read())

    model_state_dict = pickle.load(buffer)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    #model_wt =
    #keras.models.load_model(open('C:/Users/saksh/data298/application/model.pkl','rb'))
    num_classes = 55

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Load the traine weights
   
    #model.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cpu')))
    model.load_state_dict(model_state_dict)

    model = model.to(device)



    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
  

    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0,3,1,2)
    
    model.eval()

    detection_threshold = 0.70
    
    img = list(im.to(device) for im in img)
    output = model(img)

    for i , im in enumerate(img):
        boxes = output[i]['boxes'].data.cpu().numpy()
        scores = output[i]['scores'].data.cpu().numpy()
        labels = output[i]['labels'].data.cpu().numpy()

        labels = labels[scores >= detection_threshold]
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    
    sample = img[0].permute(1,2,0).cpu().numpy()
    sample = np.array(sample)
    boxes = output[0]['boxes'].data.cpu().numpy()
    name = output[0]['labels'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    names = name.tolist()
    
    return names, boxes, sample

################### PREDICT #################


def predict(image):
    names,boxes,sample = obj_detector(image)
    for i,box in enumerate(boxes):
        cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]),(0, 220, 0), 2)
        cv2.putText(sample, classes[names[i]].astype(str), (box[0],box[1]-5),cv2.FONT_HERSHEY_COMPLEX ,0.7,(220,0,0),1,cv2.LINE_AA) 
        #sample.save(r"C:\Users\saksh\data298\application\static\pred1.jpg")
        
        #plt.imshow(sample)
        img = Image.fromarray(sample, "RGB")
        
          
        img.save("C:/Users/saksh/data298/application/static/pred1.jpg" )
        

    return

    

