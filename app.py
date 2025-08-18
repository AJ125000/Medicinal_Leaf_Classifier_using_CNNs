import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Medicinal Leaf Classifier",
    page_icon="🌿",
    layout="centered"
)

# Define paths and constants
MODEL_PATH = Path("medicinal_leaf_classifier_best.pth")
LABELS_PATH = Path("labels.txt")
NUM_CLASSES = 30 # IMPORTANT: Change this to your actual number of classes (e.g., 30 or 40)

# --- MODEL AND DATA LOADING (CACHED) ---

@st.cache_resource
def load_model(model_path, num_classes):
    """Loads the pre-trained PyTorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model architecture (ResNet-50)
    model = models.resnet50(pretrained=False) # We load weights manually
    
    # --- THIS IS THE CRITICAL PART ---
    # The classifier head MUST match the one from your training script
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    # --------------------------------

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval() # Set the model to evaluation mode
    return model

@st.cache_data
def load_class_names(labels_path):
    """Loads class names from a text file."""
    with open(labels_path, "r") as f:
        # Read lines and strip any trailing newline characters
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# --- PREDICTION FUNCTION ---

def predict(model, image, class_names, device):
    """Preprocesses an image and returns the prediction and confidence."""
    # Define the image transformations (must match validation transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_label = class_names[predicted_idx.item()]
    return predicted_label, confidence.item()

# --- STREAMLIT UI ---

def main():
    st.title("🌿 Medicinal Leaf Classification")
    st.write(
        "Upload an image of a medicinal plant leaf, and the model will predict its species."
    )

    # Load model and class names
    model = load_model(MODEL_PATH, NUM_CLASSES)
    class_names = load_class_names(LABELS_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sidebar for additional information
    with st.sidebar:
        st.header("About the Model")
        st.write("This app uses a ResNet-50 deep learning model fine-tuned to classify medicinal plant leaves.")
        st.subheader("Predictable Classes")
        st.write(f"Total: {len(class_names)}")
        # Use an expander for a clean look
        with st.expander("Show all classes"):
            st.write("\n".join(f"- {name}" for name in class_names))

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a leaf image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # --- FIX: Convert from OpenCV's BGR to PIL's RGB format ---
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        # ------------------------------------------------------------
        
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify Leaf"):
            with st.spinner("Analyzing the leaf..."):
                predicted_label, confidence = predict(model, pil_image, class_names, device)
            
            st.success("Prediction Complete!")
            
            # Display results with confidence
            st.metric(label="Predicted Species", value=predicted_label)
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            
            # Optional: Add a confidence bar
            st.progress(confidence)


if __name__ == "__main__":
    # Check for required files
    if not MODEL_PATH.exists():
        st.error(f"Model file not found! Make sure '{MODEL_PATH}' is in the same directory.")
    elif not LABELS_PATH.exists():
        st.error(f"Labels file not found! Make sure '{LABELS_PATH}' is in the same directory.")
    else:
        main()