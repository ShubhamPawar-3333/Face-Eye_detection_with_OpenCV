import cv2 as cv

# Loading the cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Reading the image
img = cv.imread('test.jpg')

# Converting into grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Face detection
faces = face_cascade.detectMultiScale(gray, 1.2, 6)

# Marking the face region
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Reading the Region of Interest and converting to greyscale
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # Detecting the eyes
    eyes = eye_cascade.detectMultiScale(roi_gray)
    # Marking the eyes
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        
# Displaying the outcome
cv.imshow('img', img)
cv.waitKey()