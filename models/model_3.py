from keras.layers import Conv2D, BatchNormalization, ReLU, Add
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import  Input, Conv2D,   BatchNormalization,  ReLU ,GlobalAveragePooling2D, Dense
from keras.optimizers import Adam

def res_blocks(rec, filters): # basic residual block
    old = rec

    rec = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(rec)
    rec = BatchNormalization()(rec)
    rec = ReLU()(rec)

    rec = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(rec)
    rec = BatchNormalization()(rec)

    rec = Add()([rec, old])   # skip connection
    send = ReLU()(rec)

    return send


def res_block_with_down(x, filters):# residual block with downsampling
    old_x = Conv2D(filters, kernel_size=1, strides=2, padding="same", use_bias=False)(x)
    old_x = BatchNormalization()(old_x)

    x_neww = Conv2D(filters, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x_neww = BatchNormalization()(x_neww)
    x_neww = ReLU()(x_neww)

    x_neww = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x_neww)
    
    #
    x_neww = BatchNormalization()(x_neww)

    x_neww = Add()([x_neww, old_x])
    
    
    x_neww = ReLU()(x_neww)
    return x_neww

#this model has 3 stages of residual block 
def model_3(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Initial conv 
    x = Conv2D(32, kernel_size=3,strides= 2 ,padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual stage 1 (32 filters)
    x = res_blocks(x, 32)# --->32 filter
    x = res_blocks(x, 32)

    # Residual stage 2 (64 filters)
    x = res_block_with_down(x, 64)  #--->64 filter
    x = res_blocks(x, 64)

    # Residual stage 3 
    x = res_block_with_down(x, 128)# --->128 filter 
    x = res_blocks(x, 128)
    
    
   
    x = GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dropout(0.5)(x)  
    
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=3e-4), loss="sparse_categorical_crossentropy",   metrics=["accuracy"])

    return model
