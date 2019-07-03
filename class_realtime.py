
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#from keras.models import load_model

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Loading model weight
classifier.load_weights('EBS.h5')


# Part 3 Prediction Image filename video file that simulate realtime camrera
import numpy as np
from keras.preprocessing import image as image_utils
import time
import cv2
cap = cv2.VideoCapture('test.mp4')

while(cap.isOpened()):
    start = time.time()
    ret, frame = cap.read()
    
    if ret == True:
#       cv2.imshow('frame',frame)
        cv2.imwrite('temp.jpg',frame)
        test_image = image_utils.load_img('temp.jpg', target_size = (64, 64))
        test_image = image_utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict_on_batch(test_image)
        
        if result[0][0] == 1:
            prediction = 'REJECT'
        else:
            prediction = 'GOOD'  
        img = cv2.putText(frame,prediction,(10,150),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,255),10)
        cv2.imshow('Result',img)
        end = time.time()
        print ('I guess it is... ',prediction,' by ',round((end - start)*1000),' miliSeconds')    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyAllWindows
    else:
        break
cap.release()
cv2.destroyAllWindows