import streamlit as st
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms

# 1. Define Class Names (from your notebook)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 2. Re-create Model Architecture (matching your notebook)
def create_model():
    # Setup ViT model with default weights
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    
    # Freeze base parameters
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # Build the custom classifier head used in your notebook
    model.heads = nn.Sequential(
        nn.Linear(768, 400),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(400, 100),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(100, len(class_names))
    )
    return model, weights.transforms()

# 3. Load Model and Transforms
@st.cache_resource
def load_assets():
    model, vit_transforms = create_model()
    # Load the trained weights
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, vit_transforms

model, transform = load_assets()

# 4. Streamlit UI
st.title("🧠 Brain Tumor Detection (ViT)")
st.write("Upload an MRI image to classify the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and Predict
    img_tensor = transform(image).unsqueeze(0) #
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_idx = torch.max(probabilities, dim=0)
        
    # Show Results
    result_class = class_names[pred_idx]
    st.success(f"Prediction: **{result_class.capitalize()}**")
    st.info(f"Confidence: {conf.item()*100:.2f}%")