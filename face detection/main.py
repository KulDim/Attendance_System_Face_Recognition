import cv2
import numpy as np
import datetime


def fullFace(x, y, w, h, photo, SIZE):
    cropped = photo[y:y + w, x:h + x]
    r = float(SIZE) / cropped.shape[1]
    dim = (SIZE, int(cropped.shape[0] * r))
    return cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, photo = cap.read()

    faces = face_cascade.detectMultiScale(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), scaleFactor = 1.5,minNeighbors = 6, minSize = (30, 30))


    for (x, y, w, h) in faces:
        name = str(datetime.datetime.now().strftime("%Y %m %d %H %M %S"))
        print(name)
        cv2.imwrite('face detection/unknown/'+ str(name) + '.jpg', fullFace(x, y, w, h, photo, 400))


        cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 5) 

    cv2.imshow("video", photo)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
        
cap.release()
cv2.destroyAllWindows()