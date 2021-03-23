import imutils
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
model = load_model("modd.h5")

import cv2
import numpy as np


class Ui_MainWindow(object):
    def __init__(self):
        self.centralwidget = QtWidgets.QWidget(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1119, 860)

        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel( self.centralwidget )
        self.label.setGeometry( QtCore.QRect( 230, 100, 691, 401 ) )
        self.label.setFrameShape( QtWidgets.QFrame.StyledPanel )
        self.label.setFrameShadow( QtWidgets.QFrame.Raised )
        self.label.setText( "" )
        self.label.setPixmap( QtGui.QPixmap( "H.png" ) )
        self.label.setObjectName( "label" )

        # self.label = QtWidgets.QFrame(self.centralwidget)
        # self.label.setGeometry(QtCore.QRect(230, 100, 691, 401))
        # self.label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # self.label.setFrameShadow(QtWidgets.QFrame.Raised)
        # self.label.setObjectName("label")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(520, 710, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.op = QtWidgets.QLabel(self.centralwidget)
        self.op.setGeometry(QtCore.QRect(360, 540, 421, 111))

        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(25)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)

        self.op.setFont(font)
        self.op.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.op.setAlignment(QtCore.Qt.AlignCenter)
        self.op.setObjectName("op")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1119, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi( MainWindow )
        self.pushButton_2.clicked.connect( self.loadImage )
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.tmp = None  # Will hold the temporary image for display
        self.started = False

    def show(self, gray, faces):

        # emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        emotion_dict = {0: "Passive", 1: "Passive", 2: "Non-", 3: "Active ", 4: "Evaluative ", 5: "Non-",
                        6: "Active "}
        for (x, y, w, h) in faces:
            cv2.rectangle( self.image, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2 )
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims( np.expand_dims( cv2.resize( roi_gray, (48, 48) ), -1 ), 0 )
            prediction = model.predict( cropped_img )
            maxindex = int( np.argmax( prediction ) )
            self.op.setText(emotion_dict[maxindex] + " Listener")
            print(emotion_dict[maxindex] + " Listener")

    def loadImage(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        cam = True  # True for webcam
        if cam:
            vid = cv2.VideoCapture( 0 )
        else:
            vid = cv2.VideoCapture( 'video.mp4' )

        if self.started:
            self.started = False
            self.pushButton_2.setText( 'Start' )
            cv2.destroyAllWindows()
            MainWindow.close()
            sys.exit()
        else:
            self.started = True
            self.pushButton_2.setText( 'Exit' )

        while True:

            ret, self.image = vid.read()
            self.image = imutils.resize( self.image, height=401 )

            if not ret:
                break

            facecasc = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml' )
            gray = cv2.cvtColor( self.image, cv2.COLOR_BGR2GRAY )
            faces = facecasc.detectMultiScale( gray, scaleFactor=1.3, minNeighbors=7 )

            self.show( gray, faces )

            self.update()
            if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                break

        vid.release()
        cv2.destroyAllWindows()

    def setPhoto(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize( image, width=691 )
        frame = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
        image = QImage( frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888 )
        self.label.setPixmap( QtGui.QPixmap.fromImage( image ) )
        # self.label.show()

    def update(self):
        img = self.image
        self.setPhoto( img )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle( _translate( "MainWindow", "E-learning platform" ) )
        self.pushButton_2.setText( _translate( "MainWindow", "Start" ) )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication( sys.argv )
    app.setWindowIcon( QtGui.QIcon( 'H.png' ) )
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi( MainWindow )
    MainWindow.show()
    sys.exit( app.exec_() )
