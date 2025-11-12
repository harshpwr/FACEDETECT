import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Page configuration
st.set_page_config(
    page_title="AI Face Detection - Real vs Fake",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .real-face {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake-face {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'models/face_detector_model.h5'
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("⚠️ Model not found! Please train the model first using the training notebook.")
        return None

# Preprocess image
def preprocess_image(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (128, 128))
    
    # Normalize pixel values
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Main app
def main():
    st.title("🔍 AI Face Detection: Real vs Fake")
    st.markdown("""
    ### Upload a face image to detect if it's real or AI-generated
    This is a demonstration project using a simple CNN model trained on sample data.
    """)
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a face image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a face to analyze"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if model is not None:
            # Add predict button
            if st.button('🔎 Analyze Image'):
                with st.spinner('Analyzing...'):
                    # Preprocess and predict
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img, verbose=0)
                    
                    # Get confidence score
                    confidence = float(prediction[0][0])
                    
                    # Determine if real or fake (threshold at 0.5)
                    is_real = confidence > 0.5
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    if is_real:
                        st.markdown(f"""
                        <div class="prediction-box real-face">
                            <h2>✅ REAL FACE</h2>
                            <h3>Confidence: {confidence*100:.2f}%</h3>
                            <p>The model predicts this is a real human face.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box fake-face">
                            <h2>⚠️ AI-GENERATED FACE</h2>
                            <h3>Confidence: {(1-confidence)*100:.2f}%</h3>
                            <p>The model predicts this is an AI-generated or fake face.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed scores
                    with st.expander("📈 Detailed Confidence Scores"):
                        st.write(f"**Real Face Score:** {confidence*100:.2f}%")
                        st.write(f"**Fake Face Score:** {(1-confidence)*100:.2f}%")
                        st.progress(confidence)
        else:
            st.error("Model not available. Please train the model first!")
            st.info("📝 Run the training notebook (`training_notebook.ipynb`) to train the model.")
    
    # Information section
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a simple Convolutional Neural Network (CNN) 
        to detect whether a face image is real or AI-generated.
        
        **How it works:**
        1. Upload a face image
        2. The CNN model analyzes the image
        3. Get a confidence score and prediction
        
        **Note:** This is a proof-of-concept demo for educational purposes.
        """)
        
        st.header("🎓 College Project")
        st.write("""
        **Project Components:**
        - Streamlit Web Interface
        - CNN Model Architecture
        - Training Dataset (~200 images)
        - Jupyter Notebook for Training
        - Complete Documentation
        """)
        
        st.header("🔧 Setup")
        st.write("""
        1. Install dependencies: `pip install -r requirements.txt`
        2. Train model: Open `training_notebook.ipynb`
        3. Run app: `streamlit run app.py`
        """)

if __name__ == "__main__":
    main()
