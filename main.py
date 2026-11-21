# =====================================================
# BRAIN TUMOR DETECTION AND CLASSIFICATION USING CNN
# =====================================================

# 1.IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

#Define Dataset Paths

path = r"C:\Users\gaura\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"
train_dir =os.path.join(path, "Training")
test_dir = os.path.join(path, "Testing")

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow as tf

#load the dataset using keras image_dataset_from_directory
train_ds=keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="int",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

test_ds=keras.utils.image_dataset_from_directory(
    directory=test_dir,    
    labels="inferred",
    label_mode="int",   
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Normalize the pixel values to the range [0, 1]
def normalize(image, label):
    image=tf.cast(image, tf.float32) / 255.0
    return image, label
train_ds=train_ds.map(normalize)
test_ds=test_ds.map(normalize)

# Define the CNN model architecture
model=Sequential()

#first convolutional layer
model.add(Conv2D(
    filters=32, 
    kernel_size=(3, 3),
    padding='valid',
    activation='relu', 
    input_shape=(224, 224, 3)
))

model.add(BatchNormalization())
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2,2)
))

#second convolutional layer
model.add(Conv2D(
    filters=64, 
    kernel_size=(3, 3), 
    padding='valid', 
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2,2)
))

#Third Convolution layer
model.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2)
))

# Fully Connected neural network
model.add(Flatten())
model.add(Dense(
    units=64,
    activation='relu'
))
model.add(Dropout(0.1))

#Output Layer
model.add(Dense(
    units=4,
    activation='softmax'
))

# Display model summary
model.summary()

# Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Train the Model
history=model.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds
)

# Save the trained CNN model
model.save("brain_tumor_model.keras")

print("Model saved successfully!")
