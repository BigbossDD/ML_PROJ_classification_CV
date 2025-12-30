import tensorflow as tf
import keras as keras


def model_1(input_shape, num_classes):
    
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[28,28, 1]))
   
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,padding="SAME",activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,padding="SAME", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model