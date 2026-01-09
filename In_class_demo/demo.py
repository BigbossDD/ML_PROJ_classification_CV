#from models.model_1 import model_1 
import tensorflow as tf 
import os 
from sklearn.metrics import classification_report 
from keras.preprocessing import image 
import numpy as np 
import matplotlib.pyplot as plt 


# #this will be an in class demo , where i will use the saved models 1 , 2, 3 , to evaluate them on an actual yeast picture that i 
# #will download from the internet # i will insert the image in the In_Class_Demo folder 

def DEMO(): 
 
    
    
    
    path = r'In_class_demo\Test.jpg' # rename the image target to Test 
    
    
    img = image.load_img(path, target_size=(224, 224)) 
    img_arr = image.img_to_array(img) 
    
    
    
    img_arr = img_arr[... , ::-1] # --> converting from RGB to BGR
    img_arr = np.expand_dims(  img_arr   ,  axis=0) 
    
    
   
    img_arr = img_arr / 255.0 #  -->  normalize the image 
    
    
    class_names = ['H1', 'H2', 'H3', 'H5', 'H6']
    print('H1 --> 0 ', 'H2 --> 1  ', ' H3 --> 2 ', 'H5--> 3 ', 'H6 --> 4')
    #-------------------------------------- 
    # Using Model 1
    print("#--------\nModel 1 pred --> ")
    model = tf.keras.models.load_model("model_1.h5") 
    
    pred = model.predict(img_arr)
    print("Raw predictions -->  ", pred)
    pred_class = np.argmax(pred, axis=1)
    print(f"Predicted class   =  {pred_class[0]}")

    #---
    #model 2
    print("#--------\nmodel 2 pred --> ")
    model = tf.keras.models.load_model("model_2.h5")
    pred = model.predict(img_arr)
    print("raw predictions : ", pred)
    pred_class = np.argmax(pred, axis=1)
    print(f"predicted class   =  {  pred_class[0]}")

    #---
     #model 3 --> 
     
    print("#--------\nModel 3 pred  --> ")
    model = tf.keras.models.load_model( "model_3.h5")
    pred = model.predict(img_arr)
    print("Raw predctions:  ", pred)
    pred_class = np.argmax(pred, axis=1)
    print(f"Predicted class: {pred_class[0]}")
    
    
    print("#--------\nModel4  pred  --> ")
    model = tf.keras.models.load_model( "model_4.h5")
    pred = model.predict(img_arr)
    print("Raw predctions:  ", pred)
    pred_class = np.argmax(pred, axis=1)
    print(f"Predicted class: {pred_class[0]}")
    
    
    
    
    
    
    #plt.imshow( image.array_to_img(img_array[0])) 
    #plt.axis('off' ) 
    #plt.show( ) 
    
    
    

