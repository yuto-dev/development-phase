import os
import onnxruntime as ort
import torch
import numpy as np
from PIL import Image
from models.cnn import CustomConvNet
import time

# Path to your ONNX and PyTorch models
onnx_model_path = "model.onnx"
pt_model_path = "model_weights.pt"

# Load the PyTorch model
pt_state_dict = torch.load(pt_model_path)

# Define the model architecture
pt_model = CustomConvNet(num_classes=7)
pt_model.load_state_dict(pt_state_dict) # Correctly load the state dictionary
pt_model.eval() # Set the model to evaluation mode

# Path to the test dataset
test_dataset_path = r"dataset\test"

# Initialize counters for correct and total predictions for ONNX and PyTorch
correct_predictions_onnx = 0
total_predictions_onnx = 0
correct_predictions_pt = 0
total_predictions_pt = 0

# Create a dictionary to map integers to class labels
class_label_mapping = {
    0: 'hyundai',
    1: 'lexus',
    2: 'mazda',
    3: 'mercedes',
    4: 'opel',
    5: 'toyota',
    6: 'volkswagen'
}

# Function to perform inference with ONNX Runtime and measure latency
def infer_onnx(image_tensor):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    start_time = time.time()
    outputs = session.run(None, {input_name: image_tensor})
    end_time = time.time()
    latency = end_time - start_time
    predicted_class_index = np.argmax(outputs[0])
    return class_label_mapping[predicted_class_index], latency

# Function to perform inference with PyTorch and measure latency
def infer_pt(image_tensor):
    image_tensor = torch.tensor(image_tensor).float() # Convert to PyTorch tensor
    start_time = time.time()
    output = pt_model(image_tensor) # Perform inference
    end_time = time.time()
    latency = end_time - start_time
    _, predicted_class_index = torch.max(output, 1) # Get the index of the predicted class
    return class_label_mapping[predicted_class_index.item()], latency

# Initialize counters for latency
total_latency_onnx = 0
total_latency_pt = 0

# Iterate over each subfolder (class) in the test dataset
for class_label in os.listdir(test_dataset_path):
    class_path = os.path.join(test_dataset_path, class_label)
    if os.path.isdir(class_path):
        # Iterate over each image in the class folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            # Load and preprocess the image
            input_image = Image.open(image_path).convert('RGB') # Ensure it's in RGB format
            input_image = input_image.resize((32, 32)) # Resize to the expected dimensions
            input_tensor = np.array(input_image).astype(np.float32) / 255.0 # Normalize to [0, 1]
            input_tensor = np.transpose(input_tensor, (2, 0, 1)) # Transpose to [channels, height, width]
            input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension
            
            # Perform inference with both ONNX and PyTorch models
            predicted_class_onnx, latency_onnx = infer_onnx(input_tensor)
            predicted_class_pt, latency_pt = infer_pt(input_tensor)
            
            # Update counters based on the prediction for ONNX
            total_predictions_onnx += 1
            if predicted_class_onnx == class_label:
                correct_predictions_onnx += 1
            total_latency_onnx += latency_onnx
            
            # Update counters based on the prediction for PyTorch
            total_predictions_pt += 1
            if predicted_class_pt == class_label:
                correct_predictions_pt += 1
            total_latency_pt += latency_pt

# Calculate and print the accuracy and average latency for ONNX and PyTorch models
accuracy_onnx = correct_predictions_onnx / total_predictions_onnx
accuracy_pt = correct_predictions_pt / total_predictions_pt
avg_latency_onnx = total_latency_onnx / total_predictions_onnx
avg_latency_pt = total_latency_pt / total_predictions_pt

print(f"Accuracy of the ONNX model on the test dataset: {accuracy_onnx:.7f}")
print(f"Average latency of the ONNX model: {avg_latency_onnx:.2f} seconds")
print(f"Accuracy of the PyTorch model on the test dataset: {accuracy_pt:.7f}")
print(f"Average latency of the PyTorch model: {avg_latency_pt:.2f} seconds")
