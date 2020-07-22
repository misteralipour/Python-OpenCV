import cv2
import numpy as np

# HaarCascades Loading
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print('Error while trying to open camera!')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # add gaussian blurring to frame
            blurred = cv2.GaussianBlur(frame, (15, 15), 0)

            # Add UnBlurred Face
            blurred[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

            # display frame
            cv2.imshow('Video', blurred)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()