import keras
from keras.models import Sequential
from keras import layers, optimizers
from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.datasets import mnist

# get the mnist data set and divide it into train & test data (x_ is the actual image data, y_ are the labels)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# images are 28x28 pixels with 1 channel (greyscale image = 1 channel, rbg image would have 3 channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# image data is presented as value between 0 and 255 -> bring it to a 0 - 1 value
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# preprocess the label data (there are 10 possible labels - 0 to 9)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# keras sequential model
model = Sequential()

# convolution layer with 32 features, the first layer needs to know the input shape
model.add(Conv2D(32, 5, activation='relu', input_shape=input_shape))

# pooling layer
model.add(MaxPool2D())

# convolution layer with 64 features
model.add(Conv2D(64, 5, activation='relu'))

# pooling layer
model.add(MaxPool2D())

# flatten the input
model.add(Flatten())

# fully connected layer, 10 outputs (the probability for each digit), the sum of all probabilites is equal to 1 (softmax)
model.add(Dense(10, activation='softmax'))

# weight optimization using Adam and categorical_crossentropy as loss funciton
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=1, batch_size=32)

# evaluate our model with the test mnist set
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
