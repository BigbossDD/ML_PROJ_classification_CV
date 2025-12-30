import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import (Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization , InputLayer , Flatten) 
from keras.optimizers import Adam 
def model_2(input_shape, num_classes): 
    model = Sequential() 
    model.add(InputLayer(input_shape=input_shape))
     
    model.add(Conv2D(filters= 32 , kernel_size= 3 , strides= 1 , padding='SAME', activation='relu')) 
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size= 2 )) 
    
    
    
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", activation="relu")) 
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2)) 
    
    
    
    model.add(Conv2D(filters= 128 , kernel_size= 3 , strides= 1 , padding='SAME', activation='relu')) 
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size= 2 )) 
    
    
    
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", activation="relu")) 
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2)) 
    
    
    model.add(Flatten()) 
    
    model.add(Dense(units=128, activation='relu')) 
    model.add(Dropout(0.5)) 
    
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']) 
    return model