import os
import sys

import cv2
from tqdm import tqdm

from tools.openpose import TorchOpenpose
from tools.openpose_detection import SKEL19DEF,SKEL25DEF,OpenposeDetection
from tools.util import group_bodypose,draw_bodypose

if __name__ == "__main__":
    torch_openpose = TorchOpenpose(
        model_body25="./weight/body_25.pth",
        paf_thres=0.4
    ) 

    video_file="./videos/panoptic.mp4"
    cap = cv2.VideoCapture(video_file)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_size=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file.replace('.mp4','_processed.avi'),fourcc, 30.0, (int(width),int(height)),True)
    
    for index in tqdm(range(int(frame_size))):
        ret, frame = cap.read()
        if not ret:
            break
        openpose_detection = torch_openpose.forward(
            image_path=frame,
            scale_search=[0.5,1.0,1.5,2.0],
            definition=SKEL19DEF # SKEL25DEF
        )
        poses=group_bodypose(openpose_detection)
        frame = draw_bodypose(frame, poses,openpose_detection)
        cv2.imwrite('./videos/temp.jpg',frame)
        out.write(frame)
    
    cap.release()
    out.release()