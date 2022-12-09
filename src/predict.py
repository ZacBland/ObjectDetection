
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model

import os
import random


def load_and_predict():
    model = tf.keras.models.load_model("src\\models\\12.53_07-12-22\\model_12.53_07-12-22.h5")
    correct = 0
    total = 0
    
    classes = ["cat", "dog", "drone", "person", "pikachu"]
    
    for dir in os.listdir("tests"):
        for image_path in os.listdir(os.path.join("tests", dir)):
            img = cv2.imread(os.path.join("tests", dir, image_path))
            resize = tf.image.resize(img, (224,224))
            np.expand_dims(resize, 0)

            yhat = model.predict(np.expand_dims(resize/255, 0), verbose =0)
            value = np.argmax(yhat[0])
            
            prediction = classes[value]
            
            if(prediction in dir):
                correct+=1
            
            print("Correct: {} | Perdiction: {}".format(dir, prediction))
            total += 1

    print(correct/total)

"""
classes = ["pikachu", "dog", "drone", "person", "cat"]
model = tf.keras.models.load_model("src\\models\\16.02_07-12-22\\model_16.02_07-12-22.h5")
def predict_frame(frame):
    img = frame
    resize = tf.image.resize(frame, (224,224))
    np.expand_dims(resize, 0)
    
    yhat = model.predict(np.expand_dims(resize/255, 0), verbose =0)
    value = np.argmax(yhat[0])
    
    prediction = classes[value]
    
    return prediction
    
    """
