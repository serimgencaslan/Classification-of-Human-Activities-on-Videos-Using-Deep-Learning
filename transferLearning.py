
#%%
########### PART OF GOOGLE COLAB #############
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

!mkdir -p drive
!google-drive-ocamlfuse drive

from zipfile import ZipFile
file_name="test.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')

from zipfile import ZipFile
file_name="train.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')

import sys
sys.path.insert(0,'drive/My Drive/bitirme2')

!pip install -U keras



#%%
############# LIBRARIES ###############

import keras
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense, InputLayer, LSTM, Dropout, Flatten, Reshape
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling3D, Conv3D, MaxPooling3D

from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.python.keras.layers.normalization import BatchNormalization

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizers import SGD

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle

import json, codecs

#%%
############# MODEL ############

nb_train_samples = 15581
nb_validation_samples = 6670
batch_size=32
epoch=40
train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.3)

test_datagen = ImageDataGenerator(rescale=1.0/250, validation_split=0.3)

train_generator = train_datagen.flow_from_directory('/content/train',
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 shuffle=True,
                                                 class_mode='categorical',
                                                 subset="training")

validation_generator = test_datagen.flow_from_directory('/content/train',
                                                   target_size = (224, 224),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical',
                                                   subset="validation")


model = Sequential() 

model.add(Conv2D(filters = 128, kernel_size = 3, padding='same', activation = 'relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D())#default pool_size=2 gelir
model.add(Dropout(0.5))

model.add(Conv2D(filters = 256, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(filters = 512, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
"""
model.add(Conv2D(filters = 1024, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

"""
model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
model.summary()

#model.load_weights('drive/bitirme2/weight1.hdf5')
#model.load_weights("drive/bitirme2/weight2.hdf5")

checkpoint = ModelCheckpoint('/content/drive/My Drive/bitirme2/weights/weight52.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)

history_model=model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epoch,
        callbacks=[checkpoint],
        verbose=1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

        
#model.save_weights("drive/bitirme2/weights/weight2.h5")


#%%
############# EVALUATION ##############
#evaluation
model.load_weights("/content/drive/My Drive/bitirme2/weights/weight23.h5")
score = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size, verbose = 1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# Plot training & validation accuracy values
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%%
############ TRANSFER LEARNING WITH VGG19 ############

from tensorflow.keras import optimizers
train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1.0/ 255)

img_width = 224
img_height = 224
color_type = 3

def vgg_19_model(img_width, img_height, color_type=3):
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, color_type))
    for layer in enumerate(base_model.layers):
        layer[1].trainable = False

    #flatten the results from conv block
    x = Flatten()(base_model.output)
    #add another fully connected layers with batch norm and dropout
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)

    #add another fully connected layers with batch norm and dropout
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)

    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
   

    #add logistic layer with all car classes
    predictions = Dense(10, activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

# Load the VGG19 network
print("Loading Model...")
model_vgg19 = vgg_19_model(img_width, img_height)

model_vgg19.summary()

nb_train_samples = 10535
nb_validation_samples = 4975
batch_size=32
epoch=40

train_generator = train_datagen.flow_from_directory('/content/train',
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 shuffle=True,
                                                 class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('/content/test',
                                                   target_size = (224, 224),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical')

sgd = optimizers.SGD(lr=0.0003, momentum=0.9, decay=0.01, nesterov=True)

model_vgg19.compile(loss='categorical_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
checkpoint = ModelCheckpoint('/content/drive/My Drive/bitirme2/weights/weight34.h5',save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history_vgg19 = model_vgg19.fit_generator(
                         train_generator,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = epoch,
                         callbacks=[checkpoint],
                         verbose = 1,
                         
                         validation_data = validation_generator,
                         validation_steps = nb_validation_samples // batch_size)

#class_weight='balanced',

#%%
############ EVALUATION #############
model_vgg19.load_weights("/content/drive/My Drive/bitirme2/weights/weight34.h5", by_name=True)
score = model_vgg19.evaluate_generator(validation_generator, nb_validation_samples // batch_size, verbose = 1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# Plot training & validation accuracy values
plt.plot(history_vgg19.history['accuracy'])
plt.plot(history_vgg19.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_vgg19.history['loss'])
plt.plot(history_vgg19.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%%

########### TRANSFER LEARNING WITH RESNET50 ############

from tensorflow.keras import applications
from tensorflow.keras import optimizers

train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1.0/ 255)

img_width = 224
img_height = 224
color_type = 3

def resnet(img_width, img_height, color_type=3):
    # create the base pre-trained model
    base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, color_type))
    for layer in enumerate(base_model.layers):
        layer[1].trainable = False

    #flatten the results from conv block
    x = Flatten()(base_model.output)

    #add another fully connected layers with batch norm and dropout
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)

    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
     #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)


    #add logistic layer with all car classes
    predictions = Dense(10, activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

# Load the RESNET50 network
print("Loading Model...")
model_resnet = resnet(img_width, img_height)

model_resnet.summary()

nb_train_samples = 10535
nb_validation_samples = 4975
batch_size=32
epoch=40

train_generator = train_datagen.flow_from_directory('/content/train',
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 shuffle=True,
                                                 class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('/content/test',
                                                   target_size = (224, 224),
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical')

sgd = optimizers.SGD(lr=0.005, momentum=0.9, decay=0.01, nesterov=True)

model_resnet.compile(loss='categorical_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
checkpoint = ModelCheckpoint('/content/drive/My Drive/bitirme2/weights/weight34.h5',save_best_only=True, monitor='val_loss', mode='min', verbose=1)
history_resnet = model_resnet.fit_generator(
                         train_generator,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = epoch,
                         callbacks=[checkpoint],
                         verbose = 1,
                         
                         validation_data = validation_generator,
                         validation_steps = nb_validation_samples // batch_size)

#class_weight='balanced',

#%% 
########### EVALUATION ############

model_resnet.load_weights("/content/drive/My Drive/bitirme2/weights/weight34.h5", by_name=True)
score = model_resnet.evaluate_generator(validation_generator, nb_validation_samples // batch_size, verbose = 1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Plot training & validation accuracy values
plt.plot(history_resnet.history['accuracy'])
plt.plot(history_resnet.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_resnet.history['loss'])
plt.plot(history_resnet.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
