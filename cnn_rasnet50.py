 # Using transfer learning to classify images
 
# Importing the keras libraries
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
 
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

# Load weight
# classifier.load_weights('EPB_rasnet50.h5')
 
# Fitting the CNN to the training set
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
 
image_size = 512
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
 
training_set = data_generator.flow_from_directory('dataset/training_set',
                                                  target_size=(image_size,image_size), 
                                                  batch_size=1, 
                                                  class_mode='categorical')
 
test_set = data_generator.flow_from_directory('dataset/test_set', 
                                              target_size=(image_size, image_size), 
                                              batch_size=1, 
                                              class_mode='categorical')
 
classifier.fit_generator(training_set, 
                         steps_per_epoch=562, 
                         epochs = 30, 
                         validation_data=test_set, 
                         validation_steps=140)

classifier.save_weights('EBS_rasnet50.h5')
