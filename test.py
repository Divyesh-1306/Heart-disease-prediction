import os
import numpy as np
import joblib
from PIL import Image
from sklearn.metrics import accuracy_score

# Load the model
model = joblib.load('model.pkl')

# Load the test data
TEST_PATH = r"C:\Users\Divyesh N\Downloads\heart disease prediction\ECG_DATA\test"

def load_images(path):
    images = []
    labels = []
    # Define the class names based on the folder names
    class_names = {
        "Normal Person ECG Images (284x12=3408)": 0,
        "ECG Images of Patient that have History of MI (172x12=2064)": 1,
        "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 2,
        "ECG Images of Myocardial Infarction Patients (240x12=2880)": 3
    }
    for class_name, class_label in class_names.items():
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    image = Image.open(image_path).convert('L')  # Load as grayscale
                    image = image.resize((64, 64))  # Resize for consistency
                    image = np.array(image).flatten()  # Convert to numpy array and flatten
                    images.append(image)
                    labels.append(class_label)  # Use the numerical label
                except Exception as e:
                    print(f"Error loading image {image_name}: {e}")
    return images, labels

images, labels = load_images(TEST_PATH)
images = np.array(images)
labels = np.array(labels)

# Make predictions on the test set
y_pred = model.predict(images)

# Evaluate the model
accuracy = accuracy_score(labels, y_pred)
print(f"Accuracy: {accuracy}")
