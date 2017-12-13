#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ekrem
"""

import numpy as np
from keras.models import model_from_json
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_test = np.reshape(mnist.test.images, (-1, 28, 28, 1))
y_test = mnist.test.labels

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("model_weights.h5")

classifier.compile(optimizer = 'adadelta',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

score = classifier.evaluate(X_test, y_test, batch_size=64)

print('Accuracy of the model is ',score[1]*100,'%')
print('Loss of the model is ',score[0])