import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ISSUE 2

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

# ISSUE 3

train_images = train_images / 255.0
test_images = test_images / 255.0

'''
first layer - the input layer
            - will intake an image and format the data structure in a method acceptable by the subsequent layers
            - will be a Flatten layer that intakes a multi-dimensional array and produces an array of a single dimension
            - places all the pixel data on an equal depth during input
second layer - Dense layer with 128 nodes
             - will use a Rectified Linear Unit (ReLU) activation function
             - this will output values between 0 and 1
             - methematically this behaves as f(x)=max(0,x)
third layer - Dense layer with 10 nodes
            - uses a softmax activation function
            - this will also output values between 0 and 1
            - sum of the outputs will be 1
            - excellent at outputting probabilities
'''
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
    return(prediction)

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epoch=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)

#predictions = model.predict(test_images)
#print(predictions[0])
predictor(test_images, test_labels, 20)








