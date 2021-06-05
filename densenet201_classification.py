import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


len(tf.config.list_physical_devices('GPU'))

metadata = pd.read_csv('data/Chest_xray_Corona_Metadata.csv').drop('Unnamed: 0', axis=1)
metadata.head()

image_root_path = 'data/'

train_data_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1,
    zoom_range=0.05,
    brightness_range=(-5, 5),
    validation_split=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_data_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Calculate class weights to balance data

unique_classes = metadata.Label.unique()
all_rows = metadata.Label.to_numpy()
weights = compute_class_weight('balanced', classes=unique_classes, y=all_rows)

train_images = train_data_gen.flow_from_directory(
    image_root_path + 'train',
    class_mode='binary',
    subset='training'
)

validate_images = train_data_gen.flow_from_directory(
    image_root_path + 'train',
    class_mode='binary',
    subset='validation'
)

test_images = test_data_gen.flow_from_directory(
    image_root_path + 'test',
    class_mode='binary'
)

densenet_model = DenseNet201(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

densenet_model.trainable = False
densenet_model.summary()

model = Sequential()
model.add(densenet_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)

callback = EarlyStopping(monitor='val_loss', patience=2)

model.fit(
    train_images,
    batch_size=50,
    callbacks=[callback],
    validation_data=validate_images,
    epochs=20,
    verbose=1
)





