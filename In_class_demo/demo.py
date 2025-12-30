#from models.model_1 import model_1 
import tensorflow as tf 
import os 
from sklearn.metrics import classification_report 
from keras.preprocessing import image 
import numpy as np 
import matplotlib.pyplot as plt 
MODEL_FILENAME_M1 = "model_1.h5" 
# #this will be an in class demo , where i will use the saved models 1 , 2, 3 , to evaluate them on an actual yeast picture that i 
# #will download from the internet # i will insert the image in the In_Class_Demo folder 
def DEMO(): 
#loading the saved model 1 
    model = tf.keras.models.load_model("model_1.h5") 
    #loading and preprocessing the image 
    
    img_path = r'In_class_demo\Test.jpg' 
    # path to your image 
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) 
    #converting from RGB to BGR
    img_array = img_array[..., ::-1]
    img_array = np.expand_dims(img_array, axis=0) 
    
    
    # Create batch dimension 
    img_array = img_array / 255.0 # Normalize the image 
    #predicting the class 
    
    class_names = ['H1', 'H2', 'H3', 'H5', 'H6']
    
    predictions = model.predict(img_array)
    print("Raw predictions:", predictions)
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class: {predicted_class[0]}")
    #print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
    print(f"Probabilities: {predictions}")
    
    #printing the result 
    
    
    print(f"Predicted class: {predicted_class[0]}") 
    #displaying the image 
    plt.imshow(image.array_to_img(img_array[0])) 
    plt.axis('off') 
    plt.show() 
    
    

