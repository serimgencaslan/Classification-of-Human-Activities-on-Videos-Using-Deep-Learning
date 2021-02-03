#%%
############# PART OF GOOGLE COLAB ############
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

import sys
sys.path.insert(0,'drive/Bitirme')

!pip install -q keras

#%%

########### LIBRARIES ##############

import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, LSTM, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling3D, Conv3D, MaxPooling3D

from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from keras.preprocessing import image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle

#%%
########### READING DATA ##############

tum_veri2=pd.read_csv('drive/Bitirme/frames_classes.csv')

with open ('drive/Bitirme/texts/outfile(0-9999).txt', 'rb') as fp:
    itemlist1 = pickle.load(fp)    
Xx1=np.array(itemlist1)

with open ('drive/Bitirme/texts/outfile(10000-19999).txt', 'rb') as fp:
    itemlist2 = pickle.load(fp)   
Xx2=np.array(itemlist2)

X=np.append(Xx1,Xx2, axis=0)

#tum_veri2.columns = ['index','frames_name','classes']

Y=tum_veri2.iloc[0:20000, 2:]
ohe=OneHotEncoder(categories='auto')
Y=ohe.fit_transform(Y).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.33)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)

max = X_train.max()
X_train = X_train/max
X_test = X_test/max

#%%
############# MODEL OF CONV2D #####################

model = Sequential() 

model.add(Conv2D(filters = 32, kernel_size = 3, padding='same', activation = 'relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D())#default pool_size=2 gelir
model.add(Dropout(0.8))
model.add(Conv2D(filters = 64, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.8))
model.add(Conv2D(filters = 128, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation = 'softmax'))
model.summary()
#model.load_weights('drive/Bitirme/weight(CNN-4).hdf5')
mcp_save = ModelCheckpoint('drive/Bitirme/weight(CNN-6).hdf5', save_best_only=True, monitor='val_loss', mode='min')
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), callbacks=[mcp_save], batch_size=100)


#%%
############# CROSS VALIDATION ##############

n_split=7

liste=[]
for train_index,test_index in KFold(n_split).split(X):
  X_train,X_test=X[train_index],X[test_index]
  Y_train,Y_test=Y[train_index],Y[test_index]

  model.fit(X_train, Y_train,epochs=200, callbacks=[mcp_save], batch_size=100)
  
  print('Model evaluation ',model.evaluate(X_test,Y_test))

  liste.append(model.evaluate(X_test,Y_test))


toplam=0  #cross validation için liste içindeki elemanların toplamı
for i in range(0,3):
  toplam+=liste[i][1]
  print(liste[i][1])
print(toplam)
print(toplam/n_split)

#%%
############# CONFUSION MATRIX #############

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

pred=model.predict(X_test)
pred=np.argmax(pred, axis=1)
Y_test2=np.argmax(Y_test, axis=1)

cm=confusion_matrix(Y_test2, pred)
np.set_printoptions(precision=2)
print("Confusion Matrix")
print(cm)

# normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')

print(cm_normalized)

#plt.figure()
#plot_confusion_matrix(cm, Y_test2)
#plt.show()

# classification report 
print("Classification Report:")
print()
print (classification_report(Y_test2, pred))

"""pred=model.predict(X_test)
pred=np.argmax(pred, axis=1)
Y_test2=np.argmax(Y_test, axis=1)

cm = confusion_matrix(Y_test2, pred)"""
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)

print("recall: "+ str(np.mean(recall)))
print("precision: "+ str(np.mean(precision)))

FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

print("FP: "+ str(np.mean(FP)))
print("FN: "+ str(np.mean(FN)))
print("TP: "+ str(np.mean(TP)))
print("TN: "+ str(np.mean(TN)))


# true positive rate (recall)
TPR = TP/(TP+FN)
print("Recall/True Positive Rate: "+ str(np.mean(TPR)))

# true negative rate
TNR = TN/(TN+FP) 
print("True Negative Rate: "+ str(np.mean(TNR)))

# positive predictive value (precision)
PPV = TP/(TP+FP)
print("Precision/Positive Predictive Value : "+ str(np.mean(PPV)))

# Negative predictive value
NPV = TN/(TN+FN)
print("Negative Predictive Value: "+ str(np.mean(NPV)))

# false positive rate
FPR = FP/(FP+TN)
print("Fall out/False Positive Rate: "+ str(np.mean(FPR)))

# False negative rate
FNR = FN/(TP+FN)
print("False Negative Rate: "+ str(np.mean(FNR)))

# False discovery rate
FDR = FP/(TP+FP)
print("False Discovery Rate: "+ str(np.mean(FDR)))

print()
print()

ACC = (TP+TN)/(TP+FP+FN+TN)
print("Accuracy: "+ str(np.mean(ACC)))

#%%

########### MODEL EVALUATION #############

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = 3, padding='same', activation = 'relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D()) #default pool_size=2 
model.add(Dropout(0.8))
model.add(Conv2D(filters = 64, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.8))
model.add(Conv2D(filters = 128, padding='same', kernel_size = 3, activation = 'relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.8))

model.add(Flatten())

model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation = 'softmax'))
model.load_weights('drive/Bitirme/weight(CNN-6).hdf5')
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
score,acc=model.evaluate(X_test, Y_test, verbose=2, batch_size=100)
print("score: %.2f" %(score))
print("validation acc: %.2f" %(acc))

#%%  

############### PREDICTION ################

from keras.models import load_model
import time

sayi=128

model = load_model('drive/Bitirme/weight(CNN-4).hdf5')
model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])

a = np.reshape(X_test[sayi],[1,128,128,3])
pred=model.predict(a)
#print(pred)

#print(Y_test[sayi])

liste = []

for deger in range(0,10):
    liste.append(pred[0][deger])
max=liste[0]

for i in range(0,10):
    if max < liste[i]:
        max = liste[i]



start=time.time() # calculating prediction time

# classes of predictions
if(pred[0][0]==max):
  print("Tahmin Edilen Sınıf: Brushing Teeth")
elif(pred[0][1]==max):
  print("Tahmin Edilen Sınıf: Dining")
elif(pred[0][2]==max):
  print("Tahmin Edilen Sınıf: Laughing")
elif(pred[0][3]==max):
  print("Tahmin Edilen Sınıf: Reading Newspaper")
elif(pred[0][4]==max):
  print("Tahmin Edilen Sınıf: Singing")
elif(pred[0][5]==max):
  print("Tahmin Edilen Sınıf: Sleeping")
elif(pred[0][6]==max):
  print("Tahmin Edilen Sınıf: Swimming Backstroke")
elif(pred[0][7]==max):
  print("Tahmin Edilen Sınıf: Waking Up")
elif(pred[0][8]==max):
  print("Tahmin Edilen Sınıf: Washing Hair")
elif(pred[0][9]==max):
  print("Tahmin Edilen Sınıf: Writing")


# real classes
if(Y_test[sayi][0]==1):
  print("Gerçek Sınıf: Brushing Teeth")
elif(Y_test[sayi][1]==1):
  print("Gerçek Sınıf: Dining")
elif(Y_test[sayi][2]==1):
  print("Gerçek Sınıf: Laughing")
elif(Y_test[sayi][3]==1):
  print("Gerçek Sınıf: Reading Newspaper")
elif(Y_test[sayi][4]==1):
  print("Gerçek Sınıf: Singing")
elif(Y_test[sayi][5]==1):
  print("Gerçek Sınıf: Sleeping")
elif(Y_test[sayi][6]==1):
  print("Gerçek Sınıf: Swimming Backstroke")
elif(Y_test[sayi][7]==1):
  print("Gerçek Sınıf: Waking Up")
elif(Y_test[sayi][8]==1):
  print("Gerçek Sınıf: Washing Hair")
elif(Y_test[sayi][9]==1):
  print("Gerçek Sınıf: Writing")

end=time.time()
print(end-start)

from matplotlib import pyplot as plt
import numpy as np

data=np.array(X_test[sayi])
plt.imshow(data, interpolation='nearest')
plt.show()

#%%
######## PREDICTION OF TEST DATA #######

#---------test kümesinin hepsinin tahmini---------#
from keras.models import load_model
import time

model = load_model('drive/Bitirme/weight(CNN-4).hdf5')
model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])

dogru=0
yanlis=0
prediction_class=""
actual_class=""

start=time.time() # calculating prediction time

for sayi in range(0,3600):
    a = np.reshape(X_test[sayi],[1,128,128,3])
    pred=model.predict(a)
    print(str(sayi)+ ".eleman")
    print(pred)
    print(Y_test[sayi])
    liste = []

    for deger in range(0,10):
        liste.append(pred[0][deger])
    max=liste[0]

    for i in range(0,10):
        if max < liste[i]:
            max = liste[i]

    print(max)

    #prediction classes
    if(pred[0][0]==max):
      tahmin_sinifi="Brushing Teeth"
      print("predicted: "+ str(prediction_class))
    elif(pred[0][1]==max):
      tahmin_sinifi="Dining"
      print("predicted: "+str(prediction_class))
    elif(pred[0][2]==max):
      tahmin_sinifi="Laughing"
      print("predicted: "+str(prediction_class))
    elif(pred[0][3]==max):
      tahmin_sinifi="Reading Newspaper"
      print("predicted: "+str(prediction_class))
    elif(pred[0][4]==max):
      tahmin_sinifi="Singing"
      print("predicted: "+str(prediction_class))
    elif(pred[0][5]==max):
      tahmin_sinifi="Sleeping"
      print("predicted: "+ str(prediction_class))
    elif(pred[0][6]==max):
      tahmin_sinifi="Swimming Backstroke"
      print("predicted: "+str(prediction_class))
    elif(pred[0][7]==max):
      tahmin_sinifi="Waking Up"
      print("predicted: "+str(prediction_class))
    elif(pred[0][8]==max):
      tahmin_sinifi="Washing Hair"
      print("predicted: "+str(prediction_class))
    elif(pred[0][9]==max):
      tahmin_sinifi="Writing"
      print("predicted: "+str(prediction_class))

    
    #real classes
    if(Y_test[sayi][0]==1):
      gercek_sinif="Brushing Teeth"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][1]==1):
      gercek_sinif="Dining"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][2]==1):
      gercek_sinif="Laughing"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][3]==1):
      gercek_sinif="Reading Newspaper"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][4]==1):
      gercek_sinif="Singing"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][5]==1):
      gercek_sinif="Sleeping"
      print("actual: " + str(actual_class))
    elif(Y_test[sayi][6]==1):
      gercek_sinif="Swimming Backstroke"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][7]==1):
      gercek_sinif="Waking Up"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][8]==1):
      gercek_sinif="Washing Hair"
      print("actual: "+str(actual_class))
    elif(Y_test[sayi][9]==1):
      gercek_sinif="Writing"
      print("actual: "+str(actual_class))


    if(prediction_class==actual_class):
      true=true+1
    elif(prediction_class != actual_class):
      false=false+1

    
end=time.time()
print("prediction time: ",end-start)


print("6000 data "+str(true)+" true")
print("6000 data "+str(false)+" false")


classes      = {'0': 'brushing teeth',
                '1': 'dining',
                '2': 'laughing',
                '3': 'reading newspaper',
                '4': 'singing',
                '5': 'sleeping',
                '6': 'swimming backstroke',
                '7': 'waking up',
                '8': 'washing hair',
                '9': 'writing'}
print(classes)

from keras.models import load_model
import cv2
import numpy as np

model = load_model('drive/Bitirme/weight(CNN-1).hdf5')
model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])

img = cv2.imread('drive/Bitirme/resimler/video7932frame12.jpg')
img = cv2.resize(img,(128,128))
img = np.reshape(img,[1,128,128,3])
prediction = model.predict_classes(img)
print(prediction)

print('Predicted as: {}'.format(classes.get('{}'.format(prediction[0]))))

from keras.models import load_model

sayi=969
model = load_model('drive/Bitirme/weight(CNN-4).hdf5')
model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])
a = np.reshape(X_test[sayi],[1,128,128,3])
pred=model.predict(a)
print(pred)

print(Y_test[sayi])

data=np.array(X_test[sayi])
plt.imshow(data, interpolation='nearest')
plt.show()
