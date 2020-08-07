import tensorflow as tf
f_mnist = tf.keras.datasets.fashion_mnist
(x_train,y_train ),(x_test , y_test) = f_mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
x_train , x_test = x_train/255. , x_test/255.

class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.9:
            print("\n 90% accuracy reached so cancelling the training!")
            self.model.stop_training = True

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64 ,(3,3) ,activation='relu' ,input_shape=(28,28 , 1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64 ,(3,3),activation='relu' ),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128 , activation='relu'),
    tf.keras.layers.Dense(10 , activation='softmax')
])

model.compile(loss = 'sparse_categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'])
model.summary()
model.fit(x_train , y_train , epochs=5 , callbacks=[CustomCallbacks()])
test_loss , test_accuracy = model.evaluate(x_test , y_test)
print(test_loss , test_accuracy)
