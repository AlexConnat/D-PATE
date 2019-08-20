# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import keras # Different than "from tensorflow import keras"
from keras.models import load_model
import numpy as np

# These directories will be used to store the teachers models and predictions
import os
if not os.path.exists('MODELS'):
    os.makedirs('MODELS')
if not os.path.exists('PREDICTIONS'):
    os.makedirs('PREDICTIONS')

fashion_mnist = keras.datasets.fashion_mnist # You can try with other datasets

# Load the data from the keras helper function (by default will be stored in
# home directory folder ~/.keras)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Standard shape of the (fashion) MNIST dataset : 60000 28x28 grayscale train images
# And 10000 images of test
assert train_images.shape == (60000, 28, 28)
assert len(train_labels) == 60000
assert test_images.shape == (10000, 28, 28)
assert len(test_labels) == 10000

# Normalize the pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0


# As an example we will train 20 teachers models, it means each teacher will
# train on a subset of 3000 train_images.
# We will then gather and store their predictions on the whole 10000 public
# test_images dataset.

NB_TEACHERS = 20
NB_SAMPLES = int(60000/NB_TEACHERS) # = 3000 here
train_images_for_teachers = np.zeros((NB_TEACHERS, NB_SAMPLES, 28, 28))
train_labels_for_teachers = np.zeros((NB_TEACHERS, NB_SAMPLES))

for i in range(NB_TEACHERS):
    train_images_for_teachers[i] = train_images[i*NB_SAMPLES:(i+1)*NB_SAMPLES]
    train_labels_for_teachers[i] = train_labels[i*NB_SAMPLES:(i+1)*NB_SAMPLES]

assert train_images_for_teachers[NB_TEACHERS-1].shape == (3000, 28, 28)
assert len(train_labels_for_teachers[NB_TEACHERS-1]) == 3000


# Our model will be a super simple neural netwok composed of two fully connected
# layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


teachers_models = [ model ] * NB_TEACHERS
teachers_loss_and_acc = [ (0,0) ] * NB_TEACHERS # (loss, accuracy)


for i in range(NB_TEACHERS):

    print(f'[*] Training Teacher {i}')
    teachers_models[i].fit(train_images_for_teachers[i], train_labels_for_teachers[i], epochs=3)
    teachers_models[i].save(f'MODELS/fm_model_teacher_{i}.h5') # ;print('[*] Model saved to disk')

    print()

    print(f'[*] Evaluating Teacher {i}')
    teachers_loss_and_acc[i] = teachers_models[i].evaluate(test_images, test_labels)
    print('Test loss, Test accuracy:', teachers_loss_and_acc[i])

    print()

    # For the example we will carry on predicting in this code. But once you have
    # the saved models, you can predict in another piece of code, by retrieving the
    # Keras models using the function  -  model = load_model('fm_model.h5')

    print(f'[*] Saving predictions for Teacher {i}')
    predictions = teachers_models[i].predict(test_images) # logits (probablities
    predicted_classes = np.argmax(predictions, axis=1) # actual class numbers
    predicted_classes = predicted_classes.astype('int8') # to save some storage
    np.save(f'PREDICTIONS/fm_predictions_teacher_{i}.npy', predicted_classes)
    print('Predictions saved to disk.')

    print()

print('[*] Done.')
