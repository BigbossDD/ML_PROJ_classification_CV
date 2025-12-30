# pipeline/dataloader.py

import os
import cv2
import numpy as np


def load_defungi_dataset(
    base_path,
    img_size=(224, 224),
    verbose=True
):
    images = []
    labels = []

    # H1, H2, ..., H6 (sorted for label consistency)
    class_names = sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

    if verbose:
        print("Classes found:", class_names)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, img_size)

            images.append(img)
            labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    if verbose:
        print(f"Loaded {len(images)} images")

    return images, labels, class_names
