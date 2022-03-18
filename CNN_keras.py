# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:54:16 2019

@author: user
"""

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

batch_size = 5
num_classes = 3
epochs = 5
rate = 2/8
local = 60 
none = 294 
scratch = 52
TrainDataRecord = np.zeros((50, int((local + none + scratch) * (1 - rate))),dtype=int)

# input image dimensions
img_rows, img_cols = 100, 100

def shuffle(X,Y):
    #打亂資料
    np.random.seed(1)
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    
    for i in range(int(len(Y) * (1 - rate))):
        TrainDataSetColor(Y[i], i)

    print("\n● training 資料亂數排序結果\n(黑色: local, 綠色: none, 白色: scratch)")
    TrainDataRecord_IMG = Image.fromarray(TrainDataRecord)
    plt.imshow(TrainDataRecord_IMG)
    plt.show()
    print("\n")
    
    return X, Y

def build_data():
    # the data, split between train and test sets
    x_train = np.ones(shape=(local + none + scratch, 100, 100, 3))
    y_train = np.ones(shape=(local + none + scratch))

    #train data
    for i in range(1, local + none + scratch + 1): #1~406
        if i <= local:
            img_path = 'local/' + str(i) + '.jpg' # 1~60
            img = cv2.imread(img_path)
            x_train[i - 1] = np.array(img) # 0 ~59
            y_train[i - 1] = 0 
        elif local < i and i <= none + local: #61~354
            img_path = 'none/' + str(i - local) + '.jpg' # 1~294
            img = cv2.imread(img_path)
            x_train[i - 1] = np.array(img) # 60~353
            y_train[i - 1] = 1
        elif none + local < i and i <= local + none + scratch: #355~406
            img_path = 'scratch/' + str(i - local - none) + '.jpg' # 1~52
            img = cv2.imread(img_path)
            x_train[i - 1] = np.array(img) # 354~405
            y_train[i - 1] = 2
        
    x_train, y_train = shuffle(x_train, y_train)
    
    
    X_train = x_train[:int(x_train.shape[0]*(1 - rate))]
    Y_train = y_train[:int(y_train.shape[0]*(1 - rate))]  
    X_test = x_train[int(x_train.shape[0]*(1 - rate)):]
    Y_test = y_train[int(y_train.shape[0]*(1 - rate)):]
    
    return X_train, Y_train, X_test, Y_test

def TrainDataSetColor(Tag,Position):
    #顯示train data的label和筆數
    if(Tag == 1):
        Tag = 160   #灰
    elif(Tag == 2):
        Tag = 254   #白   
    for i in range(50):
        TrainDataRecord[i][Position] = Tag

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = build_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # 使用one hot編碼
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
 
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    plt.plot(history.history['loss'])    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'train_acc', 'val_loss', 'val_acc'], loc='upper left')
    plt.show()
            
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    prediction = model.predict(x_test)

    for i in range(np.shape(prediction)[0]):
        digit = np.argmax(prediction[i])
        if digit == 0:
            pre = "local"
        elif digit == 1:
            pre = "none"
        else:
            pre = "scratch"

        if y_test[i][0] == 1:
            ans = "local"
        elif y_test[i][1] == 1:
            ans = "none"
        else:
            ans = "scratch"
        print("第 " + str(i + 1) + "筆測試資料, model分類: " + pre + " 正確答案: " + ans)
        
    model.save('my_model.h5')
