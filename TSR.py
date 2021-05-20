import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os

import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping


data = []
labels = []

classes = 43

current_path = os.getcwd()                                                      ## get the current path

## print(current_path)


for i in range(classes):
    path = os.path.join(current_path+"/Train/"+str(i))                         ##access each image path by looping through all images
    
    images = os.listdir(path)                                                  ## list out all the image available in the path --> make them as a list
    
## print(path)
## print(images)

    try:
        for img in images:
            pic = Image.open(path+'/'+img)                                    ## opening every images separately in each class
            pic = pic.resize((30,30))                                         ## resizing all images to same standard size
            pic = np.array(pic)                                               ## since the model can process only nuumerical data as array, the picture is convered to array using numpy

            data.append(pic)                                                  ## appending image data to the list
            labels.append(i)                                                  ## appending corresponding labels in another list
    except:
        print("Image not loading")

    
## print(pic)
#print(data)
#print(labels) 
    
    
data = np.array(data)                                                        ## converting the lists made to an array compatable for the model
labels = np.array(labels)
print(data.shape,labels.shape)


## splitting the data

X = data
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## One hot encoding {1,2,3...42}--> {0...1}

y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)

## Building the model


model = Sequential()

model.add(Conv2D(filters = 32,kernel_size =(5,5), activation="relu",input_shape=X_train.shape[1:]))
model.add(Conv2D(filters = 32,kernel_size =(5,5), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(rate = 0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))

model.add(Dropout(rate=0.5))
model.add(Dense(43,activation = "softmax"))


model.compile(loss="categorical_crossentropy", optimizer = "adam",metrics = ["accuracy"])

callback = EarlyStopping(monitor='val_loss',mode = "auto")

## early stopping will stop training the data with the model
## i.e. stops the iteration(epoch) when loss becomes stable 
## running more epoch than that will have no much difference to get better accuracy,
##in this case it stops at epoch 3 (I checked for val_loss here)



epoch =15
history  = model.fit(X_train,y_train,batch_size=32,epochs=epoch,validation_data=(X_test,y_test),callbacks=[callback])  
## fitting the split train data in the model and validating with split test data



model.save("test_tsr_cnn.h5")                                ##saving the model
    



## Plotting the accurary and loss (train VS validation data)


## SUBPLOT(NO OF ROW, NO OF COLS, PLOT NO)
fig = plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.plot(model.history.history["accuracy"],label = "Training Accuracy")
plt.plot(model.history.history["val_accuracy"],label = "Validation Accuracy" )
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy rate")
plt.legend()
plt.show()

fig = plt.figure(figsize = (12,6))
plt.subplot(1,2,2)
plt.plot(model.history.history["loss"],label = "Training Loss" )
plt.plot(model.history.history["val_loss"],label = "Validation Loss" )
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("loss rate")
plt.legend()
plt.show()



## testing accuracy using test dataset

y_test_check = pd.read_csv("../TSR/Test.csv")                 ## Reading the Test.csv file
# y_test_check.head()

labels_check = y_test_check["ClassId"].values
#labels_check

imgs_check = y_test_check["Path"].values
imgs_check

data_check =[]

for j in imgs_check:

    new_pic = Image.open(j)
    result_img = new_pic.resize((30,30))
    result_img = np.array(result_img)
    data_check.append(result_img)
        
X_test_check = np.array(data_check)
pred = model.predict_classes(X_test_check)                  ## predicting classes for X_test data --> data of new images from Test.csv file



print("Accuracy Score --> " ,sm.accuracy_score(labels_check,pred)*100,"\n")
print("Classification Report \n\n ", sm.classification_report(labels_check,pred))

## Getting the accurary and classification report. Classification show for 43 classes, so this step can be skipped if we dont want to know how well each class imgs are train.
## hash it out.

model.save("traffic_sign_classifier.h5")                  ## Saving the complete model
    
