import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    # print(model_history.history.keys())
    fig, axs = plt.subplots( 1, 2, figsize=(15, 5) )
    # summarize history for accuracy
    axs[0].plot( range( 1, len( model_history.history['accuracy'] ) + 1 ), model_history.history['accuracy'] )
    axs[0].plot( range( 1, len( model_history.history['val_accuracy'] ) + 1 ), model_history.history['val_accuracy'] )
    axs[0].set_title( 'Model Accuracy' )
    axs[0].set_ylabel( 'Accuracy' )
    axs[0].set_xlabel( 'Epoch' )
    axs[0].set_xticks( np.arange( 1, len( model_history.history['accuracy'] ) + 1 ),
                       len( model_history.history['accuracy'] ) / 10 )
    axs[0].legend( ['train', 'val'], loc='best' )
    # summarize history for loss
    axs[1].plot( range( 1, len( model_history.history['loss'] ) + 1 ), model_history.history['loss'] )
    axs[1].plot( range( 1, len( model_history.history['val_loss'] ) + 1 ), model_history.history['val_loss'] )
    axs[1].set_title( 'Model Loss' )
    axs[1].set_ylabel( 'Loss' )
    axs[1].set_xlabel( 'Epoch' )
    axs[1].set_xticks( np.arange( 1, len( model_history.history['loss'] ) + 1 ),
                       len( model_history.history['loss'] ) / 10 )
    axs[1].legend( ['train', 'val'], loc='best' )
    fig.savefig( 'plot.png' )
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 150

# train_datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,rescale=1. / 255)
# val_datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,rescale=1. / 255)
train_datagen = ImageDataGenerator( rescale=1. / 255 )
val_datagen = ImageDataGenerator( rescale=1. / 255 )

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical' )

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical' )

# Create the model
model = Sequential()

model.add( Conv2D( 32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1) ) )
model.add( Conv2D( 64, kernel_size=(3, 3), activation='relu' ) )
model.add( MaxPooling2D( pool_size=(2, 2) ) )
model.add( Dropout( 0.25 ) )

model.add( Conv2D( 128, kernel_size=(3, 3), activation='relu' ) )
model.add( MaxPooling2D( pool_size=(2, 2) ) )
model.add( Conv2D( 128, kernel_size=(3, 3), activation='relu' ) )
model.add( MaxPooling2D( pool_size=(2, 2) ) )
model.add( Dropout( 0.25 ) )

model.add( Flatten() )
model.add( Dense( 1024, activation='relu' ) )
model.add( Dropout( 0.5 ) )
model.add( Dense( 7, activation='softmax' ) )

# Layers involved in the model
model.summary()


def show():
    model.load_weights( 'model.h5' )

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL( False )

    emotion_dict = {0: "Passive", 1: "Passive", 2: "Non-", 3: "Active ", 4: "Evaluative ", 5: "Non-", 6: "Active "}

    # start the webcam feed
    cap = cv2.VideoCapture( 0 )
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml' )
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        faces = facecasc.detectMultiScale( gray, scaleFactor=1.3, minNeighbors=7 )

        for (x, y, w, h) in faces:
            cv2.rectangle( frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2 )
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims( np.expand_dims( cv2.resize( roi_gray, (48, 48) ), -1 ), 0 )
            prediction = model.predict( cropped_img )
            maxindex = int( np.argmax( prediction ) )
            cv2.putText( frame, emotion_dict[maxindex]+" Listener", (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                         (255, 255, 255), 2, cv2.LINE_AA )
        cv2.imshow('video' , cv2.resize( frame, (640, 480), interpolation=cv2.INTER_CUBIC ) )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    cap.release()
    cv2.destroyAllWindows()


from pathlib import Path

my_file = Path("model.h5")
if my_file.is_file():
    pass

else:
    # If you want to train the same model or try other models, go for this
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
    model.compile( loss='categorical_crossentropy', optimizer=Adam( lr=0.0001, decay=1e-6 ), metrics=['accuracy'] )
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size )
    model.save_weights( 'model.h5' )
    plot_model_history( model_info )
show()
