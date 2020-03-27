# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second Convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))      # to find the probabilty of the output

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN into the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
                    
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
       
classifier.fit_generator(training_set,
                        steps_per_epoch = (8000 / 32),
                        epochs = 25,
                        validation_data = test_set,
                        validation_steps = (2000 / 32)
                        )

# Save model after training with dataset
classifier.save('model.m1')
#=======================================================================================
# ========================END OF MODEL BUILDING=========================================
#=======================================================================================

'''
# Predict image using trained model (either this or below)
from skimage.io import imread
from skimage.transform import resize
import numpy as np
 
class_labels = {v: k for k, v in training_set.class_indices.items()}
 
img = imread('dataset/single_prediction/cat3.jpg') #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
 
if(np.max(img)>1):
    img = img/255.0
 
prediction = classifier.predict_classes(img)
 
print(class_labels[prediction[0][0]])
'''


#=======================================================================================
# ==============================RUN BUILT MODEL=========================================
#=======================================================================================

# Loading saved model for future uses
from keras.models import load_model
classifier = load_model('CNN_Model.m1')

# Predict image using trained model (either this or above)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat4.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_classes(test_image)

if result[0][0] == 1:
    prediction = 'dog'
    print('This image contains a dog.')
else:
    prediction = 'cat'
    print('This image contains a cat.')



