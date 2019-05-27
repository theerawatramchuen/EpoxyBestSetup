# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense

# Initialise the number of classes
num_classes = 2
 
# Build the model
classifier = Sequential()
classifier.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
classifier.add(Dense(num_classes, activation='softmax'))
 
# Say not to train first layer (ResNet) model. It is already trained
classifier.layers[0].trainable = False
 
# Compiling the CNN
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Loading model weight
start = time.time()
classifier.load_weights('EPB_rasnet50.h5')

#Prediction Image filename cat_or_dog.jpg
from keras.preprocessing import image as image_utils
test_image = image_utils.load_img('dataset/single_prediction/sample.jpg', target_size = (64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

print ('Good ',result[0][1]*100.0,'%')
print ('Reject ',result[0][0]*100.0,'%')

end = time.time()
print('Prediction time is',(end - start),' Seconds')
