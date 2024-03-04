import torch
from models.cnn import CustomConvNet

# Load the trained model
model = CustomConvNet(num_classes=7)
model.load_state_dict(torch.load('model_weights.pt'))
model.eval() # Set the model to evaluation mode

# Prepare a dummy input tensor
dummy_input = torch.randn(1, 3, 32, 32)

# Export the model to an ONNX file
onnx_filename = 'model.onnx'
torch.onnx.export(model, dummy_input, onnx_filename)

print(f"Model exported to {onnx_filename}")
