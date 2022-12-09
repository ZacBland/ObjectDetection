import unittest

import os

import sys
import cv2
import torch

sys.path.append("..")
import src
from src.video import split_video

class TestVideoMethods(unittest.TestCase):
    
    def test_video(self):
        
        frames = split_video("tests/test_video.mp4")
        self.assertEqual(len(frames), 274)
    

class TestModelMethods(unittest.TestCase):
    
    def test_load_model(self):
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models\\yolov5s.pt', force_reload=True)
        
        self.assertIsNotNone(model)
        
        
        
        
if __name__ == '__main__':
    frames = split_video("tests/test_video.mp4")
    print(len(frames))