# 🌊 Water Pollution Detection using CNN - Week 1



# Step 1: Import Libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Step 2: Define Dataset 



# Step 3: Load and Preprocess Images

datagen = ImageDataGenerator(
    rescale=1./255,          # normalize pixel values
    validation_split=0.2     # 80% training, 20% validation
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Step 4: Visualize Sample Images

sample_images, sample_labels = next(train_data)

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(sample_images[i])
    plt.title("Clean" if sample_labels[i]==0 else "Polluted")
    plt.axis('off')
plt.show()


# Step 5: CNN Model Plan (Week 2 Preparation)

print(" Week 1 completed: Data collected, preprocessed, and ready for CNN model training in Week 2.")
