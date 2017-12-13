#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ekrem
"""

# Importing numpy and matplotlib

import numpy as np
import matplotlib.pyplot as plt
import time

current_milli_time = lambda: int(round(time.time()))

# Downloading and scaling the data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train = np.reshape(mnist.train.images, (55000, 28, 28, 1))

y_train = mnist.train.labels

X_test = np.reshape(mnist.test.images, (10000, 28, 28, 1))

y_test = mnist.test.labels

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Building the CNN model
classifier = Sequential()

classifier.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(optimizer = 'adadelta', 
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# Data augmentation and training
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=10)

training_set = train_datagen.flow(X_train,
                                  y_train,
                                  batch_size=64)  

time_before = current_milli_time() #time

training_log = classifier.fit_generator(training_set,
                                        steps_per_epoch=55000/64,
                                        epochs=10,
                                        validation_data=(X_test, y_test))

time_after = current_milli_time()
m, s = divmod(time_after-time_before, 60)
h, m = divmod(m, 60)

# Acquiring last score

score = classifier.evaluate(X_test, y_test, batch_size=64)

# Acquiring the results from history

result_training_acc = training_log.history['acc']
result_test_acc = training_log.history['val_acc']
result_training_loss = training_log.history['loss']
result_test_loss = training_log.history['val_loss']

# Plotting the results

x = np.arange(1,11,1)

# Plotting accuracy
plt.plot(x ,result_training_acc, 'bo')
plt.plot(x ,result_test_acc, 'ro')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy','Test accuracy'])
plt.show()

# Plotting losses
plt.plot(x ,result_training_loss, 'bo')
plt.plot(x ,result_test_loss, 'ro')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss','Test loss'])
plt.show()

# Saving models and weights
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model_weights.h5")
json_file.close()

"""
test_image = image.load_img('img.jpg', grayscale=True, target_size=(28, 28))
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict_classes(test_image)
"""

"""
# Loading models and weights
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("model_weights.h5")
"""

"""

classifier.fit(X_train,
               y_train,
               batch_size=32,
               epochs=1,
               validation_data=(X_test, y_test))
"""