import cv2


def split_video(filepath):
    """
    Takes in video and returns an array of frames

    Args:
        filepath (String): file path of video

    Returns:
        List[image]: List of frames from given video
    """
    capture = cv2.VideoCapture(filepath)
    frameNr = 0
    frames = []
    while(True):
        success, frame = capture.read()
        if success:
            frames.append(frame)
        else:
            break
        
        frameNr += 1
    
    capture.release()
    
    return frames

def draw_prediction(frame, prediction):
    """
    Draws prediction boxes onto video

    Args:
        frame: frame to be drawn on
        prediction: prediction box to draw

    Returns:
        frame: drawn-on frame
    """
    height, width, layers = frame.shape
    box_width = int(.1 * width)
    box_height = int(.05 * height)
    text = prediction
    cv2.rectangle(frame, (0,0), (box_width,box_height), (255, 255, 255), -1)
    cv2.putText(frame, text, (int(box_width/2)-25,int(box_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2, cv2.LINE_4)
    return frame

def combine_frames(frames, filename):
    """
    Combines frames and outputs a video to filepath

    Args:
        frames (list): frames to be combined
        filepath: filepath to be saved
    """
    height, width, layers = frames[0].shape
    size = (width, height)
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    
    for frame in frames:
        video.write(frame)
        
    video.release()
    
    print("saved to: " +filename)
        