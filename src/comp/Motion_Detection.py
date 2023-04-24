import cv2
import numpy as np


diffs = []
First_blur_frame = np.ndarray([])
Last_blur_frame = np.ndarray([])

if __name__ != "__main__":

    def get_blur(frame):

        gray_first = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#b2lb l gray scale ashan a 3rf a3ml minus ala level wahed bs msh 3
        gaus_first = cv2.GaussianBlur(gray_first,(21,21),0)
        blur_first = cv2.blur(gaus_first,(5,5))

        return blur_first

    def detect_motion(frame):
        global diffs
        global First_blur_frame
        global Last_blur_frame

        # check if array is empty
        if not Last_blur_frame.any():
            Last_blur_frame = get_blur(frame)
            return
        else:
            First_blur_frame = get_blur(frame)


        # Find the absolute difference between frames
        diff = cv2.absdiff(First_blur_frame, Last_blur_frame)


        Last_blur_frame = First_blur_frame
        # threshold between values bigger and smaller than 20
        thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
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

            
