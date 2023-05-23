import cv2
from time import time

time1 = 0

def get_fps(frame):
    global time1 
    global time2

    time2 = time()

    if(time2 - time1)> 0:
        frames_per_second = 1.0 / (time2- time1)
        # cv2.putText(frame,
        #             "FPS: {}".format(int(frames_per_second)),
        #             (50,100),
        #         cv2.FONT_HERSHEY_PLAIN, 2,
        #         (0,255,0), 3)                    
        time1 = time2   
    return int(frames_per_second) 