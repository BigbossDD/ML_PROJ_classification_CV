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
from models.model_2 import model_2
from models.model_3 import model_3
from models.model_4 import model_4

#--> names for the weights files , those will not go to git too big 
MODEL_FILENAME_M1 = "model_1.h5"
MODEL_FILENAME_M2 = "model_2.h5"
MODEL_FILENAME_M3 = "model_3.h5"
MODEL_FILENAME_M4 = "model_4.h5"
chosice_for_mode = True  # True --> train // False-> display stats

def main(model_num):
    #getting the data from this path --> C:\Users\USER\OneDrive\Desktop\PSUT\ML\ML_PROJ_classification_CV\defungi
    #my data type is JPG
    data_path = r"defungi"
    images, labels, class_names = load_defungi_dataset(data_path, img_size=(224, 224), verbose=True)
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    #print(images[0])  # Print the first image array
    #print("Unique labels:", np.unique(labels))  # Print unique labels --> they are [0 , 1 , 2 ,3 ,4  ]
    
    

    #---------------------------
   
    images = images / 255.0 # --> norm for the images
    
    
    #-----------------------
    #splitting the data into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)
    #validation and test sets from temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    
    
    
    #seeing one  imag
    '''
    plt.imshow(X_train[0])
    plt.title(class_names[y_train[0]])
    
    plt.show()
          '''  
          
          
# models section -->>
#---------------------------
    # i will be using GPU to fasten the training process

    
        #model_1 implementation on the 
    in_shape= X_train.shape[1:]  
    len_of_class=len( class_names)  # len of classes that the modle will classify 
    if model_num == 1 : 
        if os.path.exists(MODEL_FILENAME_M1) and not chosice_for_mode:
            print(f"loading model --> {MODEL_FILENAME_M1} ")
            model = tf.keras.models.load_model(MODEL_FILENAME_M1)
            
            
            # a classification report on test set
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print("classification report for model 1-->")
            print(classification_report(y_test, y_pred, target_names=class_names))
            
        else:
            print("training model --> ")
            with tf.device('/GPU:0'):  # optional: TF usually auto-chooses GPU
                model = model_1(in_shape, len_of_class)
                model.summary()
                
                history = model.fit(X_train, y_train,epochs=30,batch_size=32,validation_data=(X_val, y_val))
                
                
                model.save(MODEL_FILENAME_M1)#saving weights
                print(f"Model saved to file with name --> {MODEL_FILENAME_M1}")
                
                #classification report on test -->
                y_pred_probs = model.predict(X_test)
                y_pred = np.argmax(y_pred_probs, axis=1)
                print("Classification Report for Model 1:")
                print(classification_report(y_test, y_pred, target_names=class_names))
                
    
    
    #------------------------------------
    #model_2
    elif model_num == 2 : 
        if os.path.exists(MODEL_FILENAME_M2) and not chosice_for_mode:
            print(f"loading model {MODEL_FILENAME_M2} ")
            model = tf.keras.models.load_model(MODEL_FILENAME_M2)
            # f1 score and classification report on test set
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print("classification report for model 2  :")
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print("Training model...")
            with tf.device('/GPU:0'):  # optional: TF usually auto-chooses GPU
                model = model_2(in_shape, len_of_class)
                model.summary()
                history = model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val, y_val)
                )
                model.save(MODEL_FILENAME_M2) #saving weights
                print(f"model saved to {MODEL_FILENAME_M2}")
                
                
                
                # f1 score and classification report 
                y_pred_probs = model.predict(X_test)
                y_pred = np.argmax(y_pred_probs, axis=1)
                print("classification report for model 2:")
                print(classification_report(y_test, y_pred, target_names=class_names))
    #------------------------------------
    
    #model_3
    elif model_num == 3 : 
        if os.path.exists(MODEL_FILENAME_M3) and not chosice_for_mode:
            print(f"loading model from {MODEL_FILENAME_M3}")
            model = tf.keras.models.load_model(MODEL_FILENAME_M3)
            # f1 score and classification report 
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print("classification report for model 3 ")
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print("Training mode --> ")
            #with tf.device('/GPU:0'):  # -->old thing i tried to use 
            model = model_3(in_shape, len_of_class)
            model.summary()
            history = model.fit(X_train, y_train,epochs=20,batch_size=32,validation_data=(X_val, y_val))
            model.save(MODEL_FILENAME_M3)
            print(f"model saved ---> {MODEL_FILENAME_M3}")

                
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            model.save(MODEL_FILENAME_M3)
            print(f"Model saved to {MODEL_FILENAME_M3}")
                # f1 score and classification report on test set
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print("classification report  model 3:")
            print(classification_report(y_test, y_pred, target_names=class_names))
            #------------------------------------
    
    #model_4
    elif model_num == 4 : 

        if os.path.exists(MODEL_FILENAME_M4) and not chosice_for_mode:
            print(f"loading model from {MODEL_FILENAME_M4}")
            model = tf.keras.models.load_model(MODEL_FILENAME_M4)
            # f1 score and classification report 
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print("classification report for model 4: ")
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print("Training mode --> ")
            model = model_4(in_shape, len_of_class)
            model.summary()
            history = model.fit(X_train, y_train,epochs=20,batch_size=32,validation_data=(X_val, y_val) )
            model.save(MODEL_FILENAME_M4)
            print(f"model saved ---> {MODEL_FILENAME_M4}")

                
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            model.save(MODEL_FILENAME_M4)
            print(f"Model saved to {MODEL_FILENAME_M4}")
            #report:
            print("classification report for  model 4:")
            print(classification_report(y_test, y_pred, target_names=class_names))
        
    
    
    

    
    

if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main(4) #--> for training and deisplay classifica report // also send in it the model number to train or check its score 
    #DEMO() # --> for demo 
    

    print("completed!!!")
