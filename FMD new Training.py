#!/usr/bin/env python
# coding: utf-8

# In[1]:


# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# In[ ]:


import tensorflow as tf


# In[ ]:


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detectorch.model", help="path to output face mask detector model")
args, unknown = ap.parse_known_args()


# In[ ]:


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 8
BS = 10


# In[ ]:


from imutils import paths

print("[INFO] loading images...")
imagePaths = list(paths.list_images(r"E:\Hope AI\1.Tamil-20230710T103356Z-001\1.Tamil\Week11-Deep Learning Module\FMD Project\dataset\dataset"))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# In[ ]:


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)


# In[ ]:


# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# In[ ]:


# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))


# In[ ]:


# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# In[ ]:


# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)


# In[ ]:


# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


# In[ ]:


get_ipython().system('pip install model')


# In[ ]:


# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


# In[ ]:


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# In[ ]:


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)


# In[ ]:


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)


# In[ ]:


# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))


# In[ ]:


# Set the desired argument values directly
args = argparse.Namespace(model="model.h5")

# The rest of your code...

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args.model, save_format="h5")


# # plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")  # Modify this line
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")  # Modify this line
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")
# 

# In[ ]:


import matplotlib.pyplot as plt

# Load TensorBoard extension (if not already loaded)
get_ipython().run_line_magic('load_ext', 'tensorboard')

# Start TensorBoard (use the correct log_dir)
get_ipython().run_line_magic('tensorboard', '--logdir ./logs')

# Load the data from TensorBoard logs
from tensorboard.backend.event_processing import event_accumulator

event_acc = event_accumulator.EventAccumulator('./logs')
event_acc.Reload()

# List available metrics
print(event_acc.Tags())

# Extract training and validation loss and accuracy
training_loss = event_acc.Scalars('train_loss')
validation_loss = event_acc.Scalars('val_loss')
training_accuracy = event_acc.Scalars('train_accuracy')
validation_accuracy = event_acc.Scalars('val_accuracy')

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot([scalar.step for scalar in training_loss], [scalar.value for scalar in training_loss], label='train_loss')
plt.plot([scalar.step for scalar in validation_loss], [scalar.value for scalar in validation_loss], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot([scalar.step for scalar in training_accuracy], [scalar.value for scalar in training_accuracy], label='train_accuracy')
plt.plot([scalar.step for scalar in validation_accuracy], [scalar.value for scalar in validation_accuracy], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




