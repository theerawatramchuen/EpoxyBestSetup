import os
import cv2 as cv
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

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

import numpy as np
import time
from keras.preprocessing import image as image_utils
import matplotlib as plt

strpath = "C:/Users/Administrator/Desktop/PyTest/test_set/rejects/"
#"C:\Users\Administrator\Desktop\PyTest\Allimg"

print("path image : "+strpath)

list_imgfile = []

for root, directory, file in os.walk(strpath):
          for file_selected in file:
                    if '.jpg' in file_selected:
                              list_imgfile.append(root+file_selected)

cv.namedWindow('imageRun',cv.WINDOW_FREERATIO)
img = cv.imread(list_imgfile[0],1)
cv.imshow('imageRun',img)
i = 0
qty_good = 0
qty_reject = 0
for f in list_imgfile:
          i=i+1
          start = time.time()
          img = cv.imread(f,1)
          test_image = image_utils.load_img(f, target_size = (64, 64))
          test_image = image_utils.img_to_array(test_image)
          test_image = np.expand_dims(test_image, axis = 0)
          result = classifier.predict_on_batch(test_image)
          prediction = 'REJECT'
          if result[0][0] == 1:
                    prediction = 'REJECT'
                    qty_reject = qty_reject + 1
          else:
                    prediction = 'GOOD'
                    qty_good = qty_good + 1 
          img = cv.putText(img,prediction,(10,20),cv.FONT_HERSHEY_SIMPLEX,0.5,(25,255,0),1)
          cv.imshow('imageRun',img)
          end = time.time()
          print("Process : ",str(i)," ",prediction," ",round(1000*(end-start),1),"mS")
          
          cv.waitKey(1)
          del(img)
          #sleep(1)
          #print(f)
cv.waitKey(100)
cv.destroyAllWindows()
print ("Qty Good : ", qty_good, "ea.", round((qty_good/(qty_good+qty_reject))*100,2),"%")
print ("Qty Rej  : ", qty_reject, "ea.", round((qty_reject/(qty_good+qty_reject))*100,2),"%") 
print ('Qty TTL  : ', qty_good+qty_reject)

