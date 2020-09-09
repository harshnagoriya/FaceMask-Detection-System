import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import time
import copy
from PIL import Image
import glob
from flask import Flask, request, render_template
filepath = '/home/harsh/x.pth'
model = torch.load(filepath)

class_names = ['without_mask',
 'with_mask'
]

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #pil_image = Image.open(image)
    pil_image = image
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img
    
    

def classify_face(image):
    device = torch.device("cpu")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #im_pil = image.fromarray(image)
    #image = np.asarray(im)
    im = Image.fromarray(image)
    image = process_image(im)
    print('image_processed')
    img = image.unsqueeze_(0)
    img = image.float()

    model.eval()
    model.cpu()
    output = model(image)
    print(output,'##############output###########')
    _, predicted = torch.max(output, 1)
    print(predicted.data[0],"predicted")


    classification1 = predicted.data[0]
    index = int(classification1)
    return class_names[index]

def getit():
    mixer.init()
    sound = mixer.Sound('/home/harsh/Downloads/1.mp3')
    sound2 = mixer.Sound('/home/harsh/Downloads/0.mp3')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score=0
    thicc=2
    #faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    while(True):
        ret, frame = cap.read()
        cv2.imwrite("/home/harsh/0.png" ,frame)
        height,width = frame.shape[:2]
        label = classify_face(frame)
        if(label == 'with_mask'):
            sound2.play()
            return "<p align=\"center\" style=\"color:green; font-size: 100pt;\">Thank you for wearing mask</p>"
        else:
            sound.play()
            return "<p align=\"center\" style=\"color:red; font-size: 100pt;\">You haven't worn mask, Wear and come again</p>"
        cv2.putText(frame,str(label),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


app = Flask(__name__)

@app.route("/")
@app.route('/home')
def home():
   return getit()
