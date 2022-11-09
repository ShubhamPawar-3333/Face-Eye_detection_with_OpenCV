import cv2 as cv

# Loading the cascade file
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Capturing video from webcam. 
cap = cv.VideoCapture(0)

while True:
    # Reading the frame
    _, img = cap.read()
    # Converting to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detecting the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Marking the face region
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 6)
        # Reading the Region of Interest and converting to greyscale
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # Detecting the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Marking the eyes
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 4)
    # Displaying Outcome
    cv.imshow('img', img)
    # Escape key to stop webcam
    k = cv.waitKey(30) & 0xff
    if k==27:
        break
# Releasing the VideoCapture object
cap.release()