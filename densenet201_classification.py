#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.config import list_physical_devices
from sklearn.utils.class_weight import compute_class_weight


# In[7]:


len(list_physical_devices('GPU'))


# In[2]:


metadata = pd.read_csv('data/Chest_xray_Corona_Metadata.csv').drop('Unnamed: 0', axis=1)


# In[3]:


metadata.head()


# In[4]:


combi = list(set(zip(metadata.Label.tolist(), metadata.Label_1_Virus_category.tolist())))
combi


# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(8,8))
axs = axs.reshape(-1)
i = 0
for c in combi:
    label, cat = c[0], c[1]
    filename = (
        metadata[
            (metadata['Label'] == label) &
            ((metadata['Label_1_Virus_category'] == cat) | (metadata['Label_1_Virus_category')
        ]
        .isnull()))]
        .iloc[0]
        .X_ray_image_name
    )
    img = Image.open(f'data/train/{filename}')
    img.show()
    axs[i].imshow(img)
    axs[i].title.set_text(f'{c[0]} - {c[1]}')
    i += 1

plt.show()


# In[1]:


image_root_path = 'data/'


# In[ ]:


# move images

folder_map = {'Normal': 'class_0/', 'Pnemonia': 'class_1/'}
for row in metadata.itertuples():
    filename = row[1]
    folder = folder_map[row[2]]
    train_test = row[3].lower() +'/'
    current_location = f'{image_root_path}{train_test}{filename}'
    new_location = f'{image_root_path}{train_test}{folder}{filename}'
    os.rename(current_location, new_location)


# In[8]:


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


# In[9]:


# Calculate class weights to balance data

unique_classes = metadata.Label.unique()
all_rows = metadata.Label.to_numpy()
weights = compute_class_weight('balanced', classes=unique_classes, y=all_rows)


# In[12]:


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


# In[13]:


densenet_model = DenseNet201(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

densenet_model.trainable = False


# In[14]:


densenet_model.summary()


# In[15]:


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


# In[16]:


callback = EarlyStopping(monitor='val_loss', patience=2)


# In[ ]:


model.fit(
    train_images,
    batch_size=50,
    callbacks=[callback],
    validation_data=validate_images,
    epochs=20,
    verbose=1
)


# In[ ]:


print('hello')


# In[ ]:




