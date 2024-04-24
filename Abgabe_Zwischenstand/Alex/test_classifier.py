import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

model = models.load_model('cat_dog_classifier.keras')

path = 'cat_keras_prototype.webp'


img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img]))

# print(prediction)
index = np.round(prediction)
# print(index)

if index == 0:
    print("Picture is of a Dog")
else:
    print("Picture is of a Cat")