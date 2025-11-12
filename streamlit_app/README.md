# AI Face Detection: Real vs Fake

## 🎓 College Project Demonstration

A simple Python-based web application that uses deep learning to detect whether a face image is real or AI-generated. This project demonstrates the complete machine learning workflow from data preparation to model deployment.

## 📋 Project Overview

This project includes:
- **Streamlit Web Application**: Simple interface for uploading and analyzing face images
- **CNN Model**: Custom Convolutional Neural Network trained from scratch
- **Sample Dataset**: ~200 curated images (real and fake faces)
- **Training Notebook**: Jupyter notebook with complete training pipeline
- **Documentation**: Clear instructions for setup and usage

## 🏗️ Project Structure

```
streamlit_app/
├── app.py                    # Main Streamlit application
├── model.py                  # CNN model architecture
├── train.py                  # Training script
├── training_notebook.ipynb   # Jupyter notebook for training
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── dataset/                  # Training data
│   ├── real/                 # Real face images
│   └── fake/                 # AI-generated/fake face images
└── models/                   # Saved model weights (created after training)
    └── face_detector_model.h5
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

The dataset folder already contains sample images:
- `dataset/real/`: Real face images (~100 images)
- `dataset/fake/`: AI-generated face images (~100 images)

You can add more images to these folders if needed.

### Step 3: Train the Model

**Option A: Using Jupyter Notebook (Recommended)**
```bash
jupyter notebook training_notebook.ipynb
```
Then run all cells in the notebook.

**Option B: Using Python Script**
```bash
python train.py
```

The training will:
- Load images from the dataset folder
- Train a CNN model (20 epochs by default)
- Save the trained model to `models/face_detector_model.h5`
- Generate training history plots

### Step 4: Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Using the Application

1. **Upload Image**: Click on the file uploader and select a face image (JPG, JPEG, or PNG)
2. **Analyze**: Click the "Analyze Image" button
3. **View Results**: See the prediction (Real or Fake) with confidence score

## 🧠 Model Architecture

Simple CNN with:
- 4 Convolutional blocks with BatchNormalization and MaxPooling
- Dropout layers for regularization
- 2 Dense layers
- Sigmoid output for binary classification

**Input**: 128x128x3 RGB images  
**Output**: Probability score (0-1)
- Score > 0.5 → Real Face
- Score ≤ 0.5 → Fake Face

## 📁 Dataset Details

The sample dataset includes:
- **Real Faces**: ~100 images from public domain sources
- **Fake Faces**: ~100 AI-generated face images
- **Total**: ~200 images for training and validation

### Adding More Data

To improve model accuracy, you can add more images:
1. Place real face images in `dataset/real/`
2. Place fake/AI-generated images in `dataset/fake/`
3. Retrain the model using the notebook or script

## 🔧 Customization

### Training Parameters

Edit `train.py` or the notebook to modify:
```python
epochs = 20          # Number of training epochs
batch_size = 16      # Batch size
img_size = (128, 128) # Image dimensions
```

### Model Architecture

Edit `model.py` to modify the CNN architecture:
- Add/remove convolutional layers
- Change filter sizes
- Adjust dropout rates

## 📈 Training Results

After training, you'll see:
- Training and validation accuracy
- Training and validation loss
- Precision and recall metrics
- Training history plots saved as `training_history.png`

## 🎯 Project Goals

This project demonstrates:
- ✅ Building a web application with Streamlit
- ✅ Creating and training a CNN from scratch
- ✅ Image preprocessing and data handling
- ✅ Model evaluation and deployment
- ✅ Complete ML workflow documentation

## ⚠️ Important Notes

- This is a **proof-of-concept** for educational purposes
- Model accuracy depends on the training dataset size and quality
- For production use, you would need:
  - Larger and more diverse dataset
  - More sophisticated model architecture
  - Additional validation and testing
  - Model optimization and fine-tuning

## 🛠️ Troubleshooting

### Model not found error
- Make sure you've trained the model first (Step 3)
- Check that `models/face_detector_model.h5` exists

### No images found in dataset
- Verify that images are in `dataset/real/` and `dataset/fake/`
- Check image formats (should be .jpg, .jpeg, or .png)

### Memory errors during training
- Reduce batch_size in training script
- Reduce image dimensions
- Use fewer images for training

## 📝 Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Streamlit 1.31.0
- OpenCV 4.9.0
- See `requirements.txt` for complete list

## 🎓 For College Demonstration

This project is complete and ready for demonstration. It includes:
- ✅ Functional web interface
- ✅ Working AI model
- ✅ Training dataset
- ✅ Training notebook with explanations
- ✅ Complete documentation

## 📧 Support

For issues or questions about this project, refer to the code comments and documentation in the Jupyter notebook.

---

**Created for college project demonstration - 2025**
