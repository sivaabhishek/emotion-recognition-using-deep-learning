import imutils
import cv2
import numpy as np
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
model = load_model("modd.h5")



tmp = None  # Will hold the temporary image for display

def setPhoto( image):
    """ This function will take image input and resize it
        only for display purpose and convert it to QImage
        to set at the label.
    """
    tmp = image
    filename = "ss.png"
    cv2.imwrite( filename, image )
    cv2.destroyAllWindows()
    sys.exit()


def update(image):
    img = image
    setPhoto( img )

def show( gray, faces, image):

    # emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    emotion_dict = {0: "Passive", 1: "Passive", 2: "Non-", 3: "Active ", 4: "Evaluative ", 5: "Non-",
                    6: "Active "}
    for (x, y, w, h) in faces:
        cv2.rectangle( image, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2 )
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims( np.expand_dims( cv2.resize( roi_gray, (48, 48) ), -1 ), 0 )
        prediction = model.predict( cropped_img )
        maxindex = int( np.argmax( prediction ) )
        cv2.putText( image, emotion_dict[maxindex] + " Listener", (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (255, 255, 255), 2, cv2.LINE_AA )


def loadImage():
    """ This function will load the camera device, obtain the image
        and set it to label using the setPhoto function
    """
    started = False
    if started:
        started = False
    else:
        started = True
    cam = True  # True for webcam
    if cam:
        vid = cv2.VideoCapture( 0 )
    else:
        vid = cv2.VideoCapture( 'video.mp4' )
    vid.set(10, 180)
    while True:

        ret, image = vid.read()
        image = imutils.resize( image, height=480 )

        if not ret:
            break

        facecasc = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml' )
        gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
        faces = facecasc.detectMultiScale( gray, scaleFactor=1.3, minNeighbors=7 )

        show( gray, faces, image )

        update(image)
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    vid.release()
    cv2.destroyAllWindows()

loadImage()
