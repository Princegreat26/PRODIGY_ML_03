import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

# Step 1: Load the data
# Define the paths to the dataset
train_dir = 'train'

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = imread(img_path)
            img = rgb2gray(img)  # Convert to grayscale
            img = resize(img, (64, 64))  # Resize to 64x64 pixels
            img = img.flatten()  # Flatten the image
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load cat and dog images
cat_images, cat_labels = load_images_from_folder(os.path.join(train_dir, 'cat'), 0)
dog_images, dog_labels = load_images_from_folder(os.path.join(train_dir, 'dog'), 1)

# Combine the data
X = np.vstack((cat_images, dog_images))
y = np.hstack((cat_labels, dog_labels))

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = svm_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

# Step 5: Visualize some predictions
def visualize_predictions(images, true_labels, pred_labels, num_images=5):
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(64, 64), cmap='gray')
        plt.title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}')
        plt.axis('off')
    plt.show()

# Visualize some predictions
visualize_predictions(X_test, y_test, y_pred)