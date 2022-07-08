import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

train_path = '/content/drive/MyDrive/RUSU/Projekt/separated/train'
test_path = '/content/drive/MyDrive/RUSU/Projekt/separated/test'
valid_path = '/content/drive/MyDrive/RUSU/Projekt/separated/val'

my_batch_size = 30

vgg16_train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size=my_batch_size)
vgg16_valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size=my_batch_size)
vgg16_test_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size=my_batch_size, shuffle=False)

def plotImages(images_arr):
      fig, axes = plt.subplots(1,10,figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
imgs, labels = next(vgg16_train_batches)
plotImages(imgs)
print(labels)

vgg16_org = keras.applications.vgg16.VGG16()
vgg16_org.summary()

model_vgg16 = keras.models.Sequential()

for layer in vgg16_org.layers[:-1]:
  model_vgg16.add(layer)

model_vgg16.summary()

for layer in model_vgg16.layers:
  layer.trainable = False
  
model_vgg16.add(Dense(units=2, activation='softmax'))
model_vgg16.summary()
keras.utils.all_utils.plot_model(model_vgg16)

model_vgg16.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])

model_vgg16.fit(x=vgg16_train_batches, validation_data=vgg16_valid_batches,epochs=23, verbose=1)

predictions = model_vgg16.predict(x=vgg16_test_batches)

cm = confusion_matrix(y_true=vgg16_test_batches.classes, y_pred=np.argmax(predictions,axis=-1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
cm_plot_labels = ['normal', 'pneumonia']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels)

tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))

model.save('/content/drive/MyDrive/RUSU/Projekt/models/vgg16')