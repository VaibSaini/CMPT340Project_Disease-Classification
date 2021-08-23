
import feather
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt

import wave, os, glob
# %matplotlib inline

import librosa
import librosa.display
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
# sklearn Preprocessing
from sklearn.model_selection import train_test_split
#Keras
import keras
import warnings
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import splitfolders

dataset = pd.read_pickle('Final_Data.pkl') 
dataset_sequential = dataset.copy()

path_to_audio_files = []

for filename in glob.glob(os.path.join('', '*.wav')):
    path_to_audio_files.append(filename)
audio_files_data = pd.DataFrame(path_to_audio_files, columns = ['audio_file'])

print(path_to_audio_files)

dataset_sequential.columns = dataset_sequential.columns.map(str)

start = dataset_sequential.columns.get_loc("0") 
end = dataset_sequential.columns.get_loc("192")

data = dataset_sequential.iloc[:, start: end]

patient_diagnosis = dataset_sequential[["Patient number", "Diagnosis"]]

patient_number = np.array(dataset_sequential["Patient number"])
diagnosis = np.array(dataset_sequential["Diagnosis"])
dic={}
for i in range(patient_number.size):
  temp = {'{}'.format(patient_number[i]) : '{}'.format(diagnosis[i])}
  	# temp[patient_number[i]] = diagnosis[i]
  dic.update(temp)

for d in diagnosis:# make folders acc to diagnosis number
    pathlib.Path(f'./image_data/{d}').mkdir(parents=True, exist_ok=True)

for filename in audio_files_data.audio_file:
    if (filename[:3]) in patient_number:
      # print(dic[(filename[:3])])
      y, sr = librosa.load(filename, mono=True, duration=5)
      plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
      plt.axis('off');
      plt.savefig(f'./image_data/{dic[(filename[:3])]}/{filename[:-4]}.png')
      plt.clf()

splitfolders.ratio('./image_data', output="./Data", seed=1337, ratio=(.8, .2))

train_datagen = ImageDataGenerator(
        rescale=1./255, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        './Data/train',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical',
        shuffle = False)

test_set = test_datagen.flow_from_directory(
        './Data/val',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical',
        shuffle = False )

model = Sequential()
input_shape=(64, 64, 3)#1st hidden layer
model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))#2nd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))#3rd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))#Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))#Add fully connected layer.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))#Output layer
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()

epochs = 255
batch_size = 8
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

history = model.fit(
        training_set,
        steps_per_epoch=50,
        epochs=50,
        validation_data=test_set,
        validation_steps=165)

evaluation = model.evaluate_generator(generator=test_set, steps=50)
print('model accuracy: ' + str(evaluation[1]))

def visualize_training(history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
    
visualize_training(history)



