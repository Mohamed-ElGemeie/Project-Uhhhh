import cv2
import numpy as np


diffs = []
First_gray_frame = np.ndarray([])
Last_gray_frame = np.ndarray([])

if __name__ != "__main__":


    def detect_motion(frame):
        global diffs
        global First_gray_frame
        global Last_gray_frame

        # check if array is empty
        if not Last_gray_frame.any():
            Last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return
        else:
            First_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Find the absolute difference between frames
        diff = cv2.absdiff(First_gray_frame, Last_gray_frame)


        Last_gray_frame = First_gray_frame
        # threshold between values bigger and smaller than 20
        gaus_first = cv2.GaussianBlur(diff,(13,13),0)
        thresh = cv2.threshold(gaus_first, 10, 255, cv2.THRESH_BINARY)[1]
        # store differance between frames
        diff_num = round(np.mean(thresh),2)

        # append to list to find average differance
        if(len(diffs)<60):
            diffs.append(diff_num)
        else:
            diffs.pop(0)
            diffs.append(diff_num)
        
        diffs_mean = int(np.mean(diffs))

        cv2.putText(frame,
                        "diff: {}".format(diffs_mean),
                        (50,150),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (255), 3)
        
        return diffs_mean

            
