import cv2

# HarrCascades Loading
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

# Read Image
eye0 = cv2.imread('eye.jpg')
#eye = cv2.cvtColor(eye0, cv2.COLOR_BGR2GRAY)

# Webcam, main code
while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray,minNeighbors=15)
        for (ex, ey, ew, eh) in eyes:
            seye = img[y+ey:y+ey+eh, x+ex:x+ex+ew]
            cv2.imshow('eyes', seye)
            e = cv2.resize(eye0, (ew, eh))
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)
            img[y+ey:y+ey+eh, x+ex:x+ex+ew] = e
            cv2.imshow('webcam', img)

    #cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# realease webcam
cap.release()
cv2.destroyAllWindows()