import torch
from PIL import Image
import cv2
import numpy

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models\\yolov5s.pt', force_reload=True)  # or yolov5n - yolov5x6, custom

def get_frame(frame):
    model.classes = [0,15,16]
    results = model(frame)
    results.render()
    
    for im in results.ims:
        im_base64 = Image.fromarray(im)
        open_cv_image = numpy.array(im_base64)
        
        return open_cv_image
        
    