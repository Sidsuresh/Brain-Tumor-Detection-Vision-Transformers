# Brain Tumor Detection with Vision Transformer (ViT)

This project features a **Streamlit** web application that utilizes a fine-tuned **Vision Transformer (ViT-B/16)** model to classify brain MRI scans.

---

## 🚀 Deployed Link

1. [Streamlit Community Cloud]()
2. [Link to Dataset](https://drive.google.com/file/d/1Qkubjf-eujpoJDOuIAtbicHMoJfIAT2M/view?usp=sharing)

---

## 📋 Prerequisites
- Python 3.9+
- Git LFS (for model weight management)

---

## 🛠️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd brain-tumor-vit-app

2. **Create and Activate Virtual Environment (Git Bash)**
    ```bash
    python -m venv venv
    source venv/Scripts/activate
    
3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    
4. **Model Weights: Place your trained model.pth file in the root directory.**

5. **Run the App**
    ```bash
    streamlit run app/main.py
