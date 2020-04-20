import numpy as np
import pandas as pd
import tensorflow as tf
import os, sys, cv2, pickle
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten,Conv2D,MaxPooling2D

# reading images from directory store it into array
def create_training_data():
        training_date = []
        for categories in CATEGORIES:
            path = os.path.join(DATA_DIR,categories)
            class_num = CATEGORIES.index(categories)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
                    training_date.append([new_array,class_num])
                except:
                    pass
        return training_date

# for converting image into array
def prepare(filepath):
    training_date = []

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    new_image = new_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    new_image = tf.cast(new_image, tf.float32)
    return new_image

# where your training dataset is..
DATA_DIR = "../dataset_directory"

# storing names of the folder/categories you have it in dataset.
CATEGORIES = ['cotton_follow','cotton_leaf']

IMAGE_SIZE = 50


# creating numpy array 
data = np.asarray(create_training_data())

x_data = []
y_data = []

for x in data:        
    x_data.append(x[0])
    y_data.append(x[1])


x_data_np = np.asarray(x_data)/255.0
y_data_np = np.asarray(y_data)
pickle_out_y = open('y_data_np','wb')
pickle_out_x = open('x_data_np', 'wb')
pickle.dump(y_data_np, pickle_out_y)
pickle.dump(x_data_np, pickle_out_x)
pickle_out_y.close()
pickle_out_x.close()


X_Temp = open('x_data_np','rb')
x_data_np = pickle.load(X_Temp)
Y_Temp = open('y_data_np','rb')
y_data_np = pickle.load(Y_Temp)
x_data_np = x_data_np.reshape(-1, 50, 50, 1)

# splitting dataset into training and test.
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_data_np, y_data_np, test_size=0.3,random_state=101)

# model creation
model = Sequential()
model.add(Conv2D(150, (3, 3), input_shape=x_data_np.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(75, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# Fitting data into model
model.fit(x_data_np, y_data_np, batch_size=45, epochs=20, validation_split=0.3)

# saving model
model.save('Plant.model')

# test image
filepath = '../test_image'
img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

# ploting and predict
plt.imshow(img_array)
test = model.predict([prepare(filepath)])

# out put
print(CATEGORIES[int(test[0][0])])