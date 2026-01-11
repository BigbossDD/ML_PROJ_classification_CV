import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

def DEMO():
    folder_path = 'In_class_demo'
    target_files = ['H1.jpg', 'H2.jpg', 'H3.jpg', 'H5.jpg', 'H6.jpg']
    model_files = ["model_1.h5", "model_2.h5", "model_3.h5", "model_4.h5"]
    class_names = ['H1', 'H2', 'H3', 'H5', 'H6']

    # Pre-load models once
    print("Loading models... please wait.")
    loaded_models = []
    for m_file in model_files:
        if os.path.exists(m_file):
            loaded_models.append((m_file, tf.keras.models.load_model(m_file)))
    
    # --- UPDATED PLOTTING LOGIC: 2 Rows, 3 Columns ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
    axes = axes.flatten() # Makes it easier to loop (0 to 5)

    for i, filename in enumerate(target_files):
        img_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(img_path):
            print(f"Skipping {filename}: File not found.")
            axes[i].axis('off')
            continue

        # 1. Process Image
        img = image.load_img(img_path, target_size=(224, 224))
        img_display = image.img_to_array(img) / 255.0
        
        img_arr = image.img_to_array(img)
        img_arr = img_arr[..., ::-1] # RGB to BGR conversion
        img_arr = np.expand_dims(img_arr, axis=0) / 255.0

        # 2. Get predictions from all models
        prediction_text = f"File: {filename}\n" + "-"*20 + "\n"
        
        for name, model in loaded_models:
            preds = model.predict(img_arr, verbose=0)
            class_idx = np.argmax(preds, axis=1)[0]
            conf = np.max(preds) * 100
            prediction_text += f"{name}: {class_names[class_idx]} ({conf:.1f}%)\n"

        # 3. Visualize in the grid slot
        axes[i].imshow(img_display)
        axes[i].set_title(prediction_text, loc='center', fontsize=10, fontweight='bold')
        axes[i].axis('off')

    # Hide the 6th empty subplot
    if len(target_files) < len(axes):
        for j in range(len(target_files), len(axes)):
            axes[j].axis('off')

    plt.subplots_adjust(hspace=0.5, wspace=0.3) # Space between rows and columns
    plt.show()

# Run the demo
# DEMO()