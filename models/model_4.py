from keras.layers import Conv2D, BatchNormalization, ReLU, Add 
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import  Input, Conv2D,   BatchNormalization,  ReLU ,GlobalAveragePooling2D, Dense , SpatialDropout2D , Reshape , Multiply
from keras.optimizers import Adam


def squez_and_excit_support_block(x):
    fil = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(fil //  8 , activation= 'relu')(se)
    se = Dense(fil , activation= "sigmoid")(se)
    se = Reshape(( 1 ,1, fil))(se)
    return Multiply()([x  ,se] )

def res_blocks(rec, filters): # basic residual block
    old = rec

    rec = BatchNormalization()(rec)#
    rec = ReLU()(rec)    #
    rec = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(rec)
    
    rec = BatchNormalization()(rec)
    rec = ReLU()(rec)

    rec = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(rec)
    
    #rec = BatchNormalization()(rec)

    rec = Add()([rec, old])   # skip connection
    send=squez_and_excit_support_block(rec)#
    #send = ReLU()(rec)

    return send


def res_block_with_down(x, filters):# residual block with downsampling
    old_x = Conv2D(filters, kernel_size=1, strides=2, padding="same", use_bias=False)(x)
    #old_x = BatchNormalization()(old_x)
    x =BatchNormalization()(x) #
    x = ReLU()(x)#
    x_neww = Conv2D(filters, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    
    x_neww = BatchNormalization()(x_neww)
    x_neww = ReLU()(x_neww)
    x_neww = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x_neww)
    
    #
    #x_neww = BatchNormalization()(x_neww)

    x_neww = Add()([x_neww, old_x])
    x_neww = squez_and_excit_support_block(x_neww)#
    
    #x_neww = ReLU()(x_neww)
    return x_neww

#this model has 3 stages of residual block 
def model_4(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Initial conv : 1
    x = Conv2D(32, kernel_size=3,strides= 2 ,padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x_1 = ReLU()(x)
    # added another  conv : 2 
    x_1 = Conv2D(32, kernel_size=3, strides= 1 ,padding="same", use_bias=False)(x_1)
    x_1 = BatchNormalization()(x)
    x = ReLU()(x_1)

    # Residual stage 1 (32 filters)
    x = res_blocks(x, 32)# --->32 filter
    x = res_blocks(x, 32)
    x = res_blocks(x, 32) # added an extra one to each stage  (the first one is down sampling other than the first stage )

    # Residual stage 2 (64 filters)
    x = res_block_with_down(x, 64)  #--->64 filter
    x = res_blocks(x, 64)
    x = res_blocks(x, 64)#

    # Residual stage 3 
    x = res_block_with_down(x, 128)# --->128 filter 
    x = res_blocks(x, 128)
    x = res_blocks(x, 128)#
    
    #-----added a batch norm and replaced the normal dropout with spatial one 
    x = BatchNormalization()(x)
    x =ReLU() (x)
    x = SpatialDropout2D(0.5 )(x)#started with 0.5 
     
    x = GlobalAveragePooling2D()(x)
    
    #x_finall = tf.keras.layers.Dropout(0.5)(x)  
    
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    #custom schedulaer for learning rate -->
    a =keras.optimizers.schedules.CosineDecay(initial_learning_rate= 3e-4 , decay_steps= 30_000) # this will decay the a evwe 30k step
    
    model.compile(
        optimizer=Adam(learning_rate=a), loss="sparse_categorical_crossentropy"
        ,   metrics=["accuracy"])

    return model
