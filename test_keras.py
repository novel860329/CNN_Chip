# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:28:11 2019

@author: user
"""
import tensorflow as tf
import cv2
import numpy as np

model = tf.contrib.keras.models.load_model('my_model.h5')

"""
    prediction = model.predict(x_test)

    for i in range(np.shape(prediction)[0]):
        digit = np.argmax(prediction[i])
        print(digit," ",y_test[i])
"""
img = cv2.imread('local/50.jpg')

test = np.array(img)
test = test.reshape(1, 100, 100, 3)
result = model.predict(test, batch_size = 1)

if result[0][0]:
    print("local")
elif result[0][1]:
    print("none")
elif result[0][2]:
    print("scratch")
    
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()