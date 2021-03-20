import time
import os
import cv2

while True:
    os.system("python Capture_and_save_photo.py")
    path = r'C:\Users\sivaa\PycharmProjects\Projects\Emotion Detection\src\ss.png'
    image = cv2.imread( path )
    cv2.imshow("image",image)
    cv2.waitKey( 0 )
    cv2.destroyAllWindows()
    time.sleep(50)
