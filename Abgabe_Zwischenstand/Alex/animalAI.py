import matplotlib as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import models, layers

train_size = 0.85
directory_train = 'afhq/train'
batch_size = 32

class_names= sorted(os.listdir(directory_train))
print("ClassNames: ", class_names)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=directory_train,
    labels='inferred',
    image_size= (512,512),
    batch_size= 32,
    shuffle=True,
    seed=123
)

train_size = int(len(train_dataset) * train_size)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    directory='afhq/val',
    labels='inferred',
    image_size= (512,512),
    batch_size= 32,
    shuffle=True,
    seed=123
)

training_ds = train_dataset.take(train_size)
test_ds = train_dataset.skip(train_size)

for images, labels in training_ds:
    train_label = [class_names[label] for label in labels]

        
for images, labels in test_ds:
    test_label = [class_names[label] for label in labels]


model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(512,512,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(training_ds, epochs=10, validation_data=(val_dataset))

loss, accuracy = model.evaluate(test_ds)
print(f"loss:  {loss}")
print(f"accuracy: {accuracy}")

model.save('cat_dog_classifier.keras')