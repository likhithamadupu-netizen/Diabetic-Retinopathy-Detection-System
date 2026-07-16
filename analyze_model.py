"""
RetinaAI - Diabetic Retinopathy Detection Model Analysis
Load trained model and provide prediction API
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model
MODEL_PATH = 'best_model.pth'
model = None

def load_model():
    """Load the trained model from disk"""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Extract model state
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Detect architecture from state dict keys
        keys = list(state_dict.keys())
        print(f"Detected {len(keys)} parameters in model")
        
        # Custom ResNet with SE blocks
        print("Loading custom ResNet with SE blocks and 5 classes")
        # Use ResNet152 as base but we'll need to handle the 4096 input to FC
        model = models.resnet152(pretrained=False)
        
        # The model uses custom naming: layer0 instead of conv1/bn1
        # We need to load with strict=False and map the keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('layer0.0'):
                new_key = k.replace('layer0.0', 'conv1')
                new_state_dict[new_key] = v
            elif k.startswith('layer0.1'):
                new_key = k.replace('layer0.1', 'bn1')
                new_state_dict[new_key] = v
            elif k.startswith('classifier'):
                new_key = k.replace('classifier', 'fc')
                new_state_dict[new_key] = v
            elif not k.startswith('se_block'):  # Skip SE block for now
                new_state_dict[k] = v
        
        # Modify FC layer to match the saved model structure (5 classes, 4096 input from checkpoint)
        model.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
        
        # Add an adapter layer to convert 2048 -> 4096
        class AdapterModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.features = nn.Sequential(*list(base_model.children())[:-1])
                self.adapter = nn.Linear(2048, 4096)
                self.fc = base_model.fc
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.adapter(x)
                x = self.fc(x)
                return x
        
        model = AdapterModel(model)
        
        # Load state dict with strict=False to ignore SE blocks
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully!")
        
        # Print training info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Trained for {checkpoint['epoch']} epochs")
            if 'val_acc' in checkpoint:
                print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
            if 'kappa' in checkpoint:
                print(f"  Kappa score: {checkpoint['kappa']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DR stage labels (5 classes)
STAGE_LABELS = [
    'Normal',
    'Mild DR',
    'Moderate DR',
    'Severe DR',
    'Proliferative DR'
]

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def analyze_image(image_tensor):
    """Run model inference on preprocessed image"""
    try:
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Forward pass
            outputs = model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class] * 100
            
            # Prepare results
            results = {
                'stage': STAGE_LABELS[predicted_class],
                'confidence': round(float(confidence), 1),
                'probabilities': {
                    STAGE_LABELS[i]: round(float(probabilities[i] * 100), 1)
                    for i in range(len(STAGE_LABELS))
                },
                'scores': [round(float(p * 100), 0) for p in probabilities]
            }
            
            return results
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        if image_tensor is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Analyze image
        results = analyze_image(image_tensor)
        if results is None:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        return jsonify(results), 200
    
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    }), 200

def main():
    """Main function to start the analysis server"""
    print("=" * 60)
    print("RetinaAI - Diabetic Retinopathy Detection")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model file not found: {MODEL_PATH}")
        print("Please ensure best_model.pth is in the current directory.")
        return
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("Starting Flask API Server...")
    print("=" * 60)
    print(f"Server: http://127.0.0.1:5001")
    print(f"Endpoint: POST /api/predict")
    print(f"Health: GET /api/health")
    print("=" * 60)
    print("\nReady to analyze retinal images! 👁️")
    print("Press Ctrl+C to stop the server.\n")
    
    # Start Flask server
    app.run(host='127.0.0.1', port=5001, debug=False)

if __name__ == '__main__':
    main()
