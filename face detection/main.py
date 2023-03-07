import cv2
import numpy as np
import datetime

FULLSCREEN = True
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
cap = cv2.VideoCapture(0)


if FULLSCREEN:
    cv2.namedWindow("Face-recognition",cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("Face-recognition",cv2.WND_PROP_ASPECT_RATIO,cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("Face-recognition",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


def fullFace(x, y, w, h, photo, SIZE):
    cropped = photo[y:y + w, x:h + x]
    r = float(SIZE) / cropped.shape[1]
    dim = (SIZE, int(cropped.shape[0] * r))
    return cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)




time = datetime.datetime.now()
while True:
    ret, photo = cap.read()
    faces = face_cascade.detectMultiScale(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY),scaleFactor = 1.3,minNeighbors = 5)

    for (x, y, w, h) in faces:
        name = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if (datetime.datetime.now() - time) > datetime.timedelta(seconds=1):
            cv2.imwrite('unknown/'+ str(name) + '.jpg', fullFace(x, y, w, h, photo, 800))
            time = datetime.datetime.now()
        cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    cv2.putText(photo, "ENTER q TO EXIT:", (5, 10), FONT, 0.5, (0, 0, 255), 1)
    cv2.imshow("Face-recognition", photo)
    if cv2.waitKey(25) & 0xFF == ord('q'): break
        
cap.release()
cv2.destroyAllWindows()