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

train_batches = ImageDataGenerator(rescale=1/255,featurewise_center=False,  samplewise_center=False,  featurewise_std_normalization=False,  samplewise_std_normalization=False,  zca_whitening=False,  zoom_range = 0.2, vertical_flip=False).flow_from_directory(directory=train_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size=my_batch_size)
valid_batches = ImageDataGenerator(rescale=1/255,featurewise_center=False,  samplewise_center=False,  featurewise_std_normalization=False,  samplewise_std_normalization=False,  zca_whitening=False,  zoom_range = 0.2, vertical_flip=False).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size=my_batch_size)
test_batches = ImageDataGenerator(rescale=1/255,featurewise_center=False,  samplewise_center=False,  featurewise_std_normalization=False,  samplewise_std_normalization=False,  zca_whitening=False,  zoom_range = 0.2, vertical_flip=False).flow_from_directory(directory=test_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size=my_batch_size, shuffle=False)

resnet=keras.applications.resnet.ResNet50(input_shape=[224,224,3],weights='imagenet',include_top=False)

for layers in resnet.layers[:-1]:
    layers.trainable=False
    
x = keras.layers.Flatten()(resnet.output)

prediction = keras.layers.Dense(2, activation='sigmoid')(x)
model = keras.models.Model(inputs=resnet.input, outputs=prediction)

model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['acc'])

model.fit(x=train_batches, validation_data=valid_batches,epochs=23, verbose=1)

predictions = model.predict(x=test_batches)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions,axis=-1))

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
    
precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))

model.save('/content/drive/MyDrive/RUSU/Projekt/models/resnet')