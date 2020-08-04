import tensorflow as tf
from tensorflow import keras
import numpy as np

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.7:
            print('\n Reached 70% accuracy so cancelling training!')
            self.model.stop_training = True

callbacks = mycallback()
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                                 keras.layers.Dense(512 ,activation=tf.nn.relu),
                                 keras.layers.Dense(10 , activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy' , metrics=['accuracy'] )
model.fit(x_train , y_train , epochs=10,callbacks=[callbacks])
# model.predict(test_images[1])
