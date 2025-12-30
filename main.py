import os
import pandas as pd
import numpy as np
from Pipe.load_data import load_defungi_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.model_1 import model_1  
import tensorflow as tf
from sklearn.metrics import classification_report
from In_class_demo.demo import DEMO


MODEL_FILENAME_M1 = "model_1.h5"
FORCE_RETRAIN = False  # Set True if you want to retrain and overwrite the model
MODEL_FILENAME_M2 = "model_2.h5"
MODEL_FILENAME_M3 = "model_3.h5"


def main():
    #getting the data from this path --> C:\Users\USER\OneDrive\Desktop\PSUT\ML\ML_PROJ_classification_CV\defungi
    #my data type is JPG
    data_path = r"defungi"
    images, labels, class_names = load_defungi_dataset(data_path, img_size=(224, 224), verbose=True)
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    #print(images[0])  # Print the first image array
    print("Unique labels:", np.unique(labels))  # Print unique labels
    
    

    #---------------------------
    #normalizing the images
    images = images / 255.0
    
    
    #-----------------------
    #splitting the data into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)
    #validation and test sets from temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    #visualizing some samples from the dataset
    #visualizing one image
    plt.imshow(X_train[0])
    plt.title(class_names[y_train[0]])
    plt.axis("off")
    plt.show()
            
#---------------------------
    # i will be using GPU to fasten the training process

    
        #model_1 implementation on the 
    input_shape = X_train.shape[1:]  
    num_classes = len(class_names)  # len of classes that the modle will classify 
    
    if os.path.exists(MODEL_FILENAME_M1) and not FORCE_RETRAIN:
        print(f"Loading model from {MODEL_FILENAME_M1}...")
        model = tf.keras.models.load_model(MODEL_FILENAME_M1)
    else:
        print("Training model...")
        with tf.device('/GPU:0'):  # optional: TF usually auto-chooses GPU
            model = model_1(input_shape, num_classes)
            model.summary()
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_val, y_val)
            )
            model.save(MODEL_FILENAME_M1)
            print(f"Model saved to {MODEL_FILENAME_M1}")

    
    #------------------------------------
    #model_2
    
    #model.save(MODEL_FILENAME_M2)
    #print(f"Model saved to {MODEL_FILENAME_M2}")
    #------------------------------------
    #model_3
    
    
    
    #model.save(MODEL_FILENAME_M3)
    #print(f"Model saved to {MODEL_FILENAME_M3}")
    
    

if __name__ == "__main__":
   # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #main()
    DEMO()
    print("completed.")
