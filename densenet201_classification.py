import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


def prepare_data(ds, shuffle=False, augment=False):

    ds = ds.map(lambda x, y: (preprocess_input(x), y))

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    
    assert len(tf.config.list_physical_devices('GPU')) >= 1

    batch_size = 50
    input_shape = (224, 224, 3)
    AUTOTUNE = tf.data.AUTOTUNE

    metadata = pd.read_csv('data/Chest_xray_Corona_Metadata.csv').drop('Unnamed: 0', axis=1)
    metadata.head()

    image_root_path = 'data/'

    train_data = image_dataset_from_directory(
        'data/train/',
        labels='inferred',
        batch_size=50,
        image_size=input_shape[:2],
        seed=0,
        validation_split=0.2,
        subset='training'
    )

    validation_data = image_dataset_from_directory(
        'data/train/',
        labels='inferred',
        batch_size=50,
        image_size=input_shape[:2],
        seed=0,
        validation_split=0.2,
        subset='validation'
    )

    test_data = image_dataset_from_directory(
        'data/test/',
        labels='inferred',
        image_size=input_shape[:2]
    )

    data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomHeight(0.1),
    layers.experimental.preprocessing.RandomWidth(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    ])

    # train_data = prepare_data(train_data, shuffle=True, augment=True)
    # validation_data = prepare_data(validation_data)
    # test_data = prepare_data(test_data)

    train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
    validation_data = validation_data.map(lambda x, y: (preprocess_input(x), y))
    test_data = test_data.map(lambda x, y: (preprocess_input(x), y))

    train_data = train_data.prefetch(buffer_size=AUTOTUNE)
    validation_data = validation_data.prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=AUTOTUNE)
    
    # Calculate class weights to balance data
    unique_classes = metadata.Label.unique()
    all_rows = metadata.Label.to_numpy()
    weights = compute_class_weight('balanced', classes=unique_classes, y=all_rows)

    densenet_model = DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    densenet_model.trainable = False
    densenet_model.summary()

    model = Sequential()
    model.add(data_augmentation)
    model.add(densenet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

    callback = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(
        train_data,
        batch_size=50,
        callbacks=[callback],
        epochs=10,
        validation_data=validation_data,
        verbose=1
    )
