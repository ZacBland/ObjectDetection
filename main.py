import cv2
import sys
import os
from src import *

from tqdm import tqdm
from yolo import get_frame

import re

if __name__ == '__main__':
    
    print("Number of arguments: ", len(sys.argv))
    print("Argument List: " + str(sys.argv))
    
    #Exit if no arguments found
    if(len(sys.argv) <= 1):
        print("No command line arguments found.")
        exit()
        
    for i in range(1, len(sys.argv)):
        try:
            filename = sys.argv[i]
            
            split_name = re.split(r"\\", filename)
            print(split_name[-1])
            
            frames = split_video(sys.argv[i])
            print("length: {0}".format(len(frames)))
            print("Split video, now saving frames...")
            frame_num = 1
            
            pbar = tqdm(total=len(frames))
            for i in range(len(frames)):
                frame_num += 1
                frames[i] = get_frame(frames[i])
                pbar.update(1)
            pbar.close()
            
            
            filename = "predicted_" + split_name[-1]
            combine_frames(frames, filename)
            
            
            
        except Exception as e:
            print(e)
            
    
            
    print("success")