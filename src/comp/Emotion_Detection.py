import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('src\comp\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\model\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)



emotion_model.load_weights("src\comp\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\model\emotion_model.h5")
print("Loaded model from disk")

face_detector = cv2.CascadeClassifier('src\comp\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\haarcascades\haarcascade_frontalface_default.xml')

if __name__ != "__main__":
    
    def detect_emotion(frame):


        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        if  len(num_faces) == 0:
            return "No face Detected"
        x=num_faces[0][0]
        y=num_faces[0][1]
        w=num_faces[0][2]
        h=num_faces[0][3]


        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        label = emotion_dict[int(np.argmax(emotion_prediction))]
        return label