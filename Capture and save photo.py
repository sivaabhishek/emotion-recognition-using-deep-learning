import imutils
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
import cv2
import numpy as np
import time

from emotions import *


class Ui_MainWindow( object ):
    def __init__(self):
        self.centralwidget = QtWidgets.QWidget( MainWindow )

    def setupUi(self, MainWindow):
        MainWindow.setObjectName( "MainWindow" )
        MainWindow.resize( 498, 522 )
        self.centralwidget.setObjectName( "centralwidget" )
        self.gridLayout_2 = QtWidgets.QGridLayout( self.centralwidget )
        self.gridLayout_2.setObjectName( "gridLayout_2" )
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName( "horizontalLayout" )
        self.label = QtWidgets.QLabel( self.centralwidget )
        self.label.setText( "" )
        self.label.setPixmap( QtGui.QPixmap( "images/H.png" ) )
        self.label.setObjectName( "label" )
        self.horizontalLayout.addWidget( self.label )
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName( "gridLayout" )
        self.horizontalLayout.addLayout( self.gridLayout )
        self.gridLayout_2.addLayout( self.horizontalLayout, 0, 0, 1, 2 )
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName( "horizontalLayout_2" )
        self.pushButton_2 = QtWidgets.QPushButton( self.centralwidget )
        self.pushButton_2.setObjectName( "pushButton_2" )
        self.horizontalLayout_2.addWidget( self.pushButton_2 )
        self.gridLayout_2.addLayout( self.horizontalLayout_2, 1, 0, 1, 1 )
        spacerItem = QtWidgets.QSpacerItem( 313, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum )
        self.gridLayout_2.addItem( spacerItem, 1, 1, 1, 1 )
        MainWindow.setCentralWidget( self.centralwidget )
        self.statusbar = QtWidgets.QStatusBar( MainWindow )
        self.statusbar.setObjectName( "statusbar" )
        MainWindow.setStatusBar( self.statusbar )

        self.retranslateUi( MainWindow )
        # self.pushButton_2.clicked.connect( self.loadImage )
        QtCore.QMetaObject.connectSlotsByName( MainWindow )

        self.tmp = None  # Will hold the temporary image for display
        self.started = False

        self.loadImage()

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
            cv2.putText( self.image, emotion_dict[maxindex] + " Listener", (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (255, 255, 255), 2, cv2.LINE_AA )

    def loadImage(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        if self.started:
            self.started = False
            self.pushButton_2.setText( 'Start' )
        else:
            self.started = True
            self.pushButton_2.setText( 'Stop' )
        model.load_weights( 'model.h5' )
        cam = True  # True for webcam
        if cam:
            vid = cv2.VideoCapture( 0 )
        else:
            vid = cv2.VideoCapture( 'video.mp4' )

        while True:

            ret, self.image = vid.read()
            self.image = imutils.resize( self.image, height=480 )

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
        filename = "ss.png"
        cv2.imwrite( filename, self.image )
        cv2.destroyAllWindows()
        sys.exit()


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
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi( MainWindow )
    MainWindow.show()
    sys.exit( app.exec_() )
