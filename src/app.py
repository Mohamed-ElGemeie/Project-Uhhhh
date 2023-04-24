import sys
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import cv2
from time import time
from comp import Pose_Detection as ps , Motion_Detection as md
from utils import fps
import numpy as np

class MainWindow(QWidget):
    def __init__(self) -> None:
        super(MainWindow , self).__init__()


        self.VBL = QVBoxLayout()
        
        # The box that holds the image
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        # the cancel button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        # the thread that holds creates the camera feed
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)


        self.setLayout(self.VBL)

    def ImageUpdateSlot(self,Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):

    ImageUpdate = pyqtSignal(QImage)
    
    def run(self):
        
        # used for the camra feed while loop
        self.ThreadActive = True

        # set camera resolution
        Capture = cv2.VideoCapture(0)
        Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        
        # for Fps tracking
        Time_start = 0
        Fps = 30

        while self.ThreadActive:
            
            ret , frame = Capture.read()
            frame = cv2.flip(frame, 1)

            TIME_PASSED = time() - Time_start

            # if the amount of time passed is more than 1/fps 
            # and camera feed is succeful
            if not (TIME_PASSED > 1. / Fps and ret):
                continue
            Time_start = time()


            # Puts the current pose's label and landmarks
            ps.detect_pose(frame)
            
            # Puts the current FPS of the camera feed
            fps.get_fps(frame)

            # Puts the amount of motion detected
            md.detect_motion(frame)


            
            # converts the image from BGR to RGB for display
            frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # to QImage
            ConvertToQtFormat = QImage(frame.data, frame.shape[1],frame.shape[0],
                                        QImage.Format.Format_RGB888)
            # sends the image to the main window
            Pic = ConvertToQtFormat.scaled(640,512, Qt.AspectRatioMode.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)


    def stop(self):
        """
        Fuction that exits the thread and stops the camera feed while loop
        """
        self.ThreadActive = False
        self.quit()
        self.wait()
        
if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())