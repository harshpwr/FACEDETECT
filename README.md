# AI Face Detection Project - College Demonstration

## 🎓 Overview

This is a complete Python-based web application that uses deep learning to detect whether a face image is real or AI-generated. Built as a college activity demonstration project.

## 📦 What's Included

### Core Components:
1. **Streamlit Web Application** - Simple, user-friendly interface for image upload and analysis
2. **CNN Model** - Custom Convolutional Neural Network built from scratch
3. **Sample Dataset** - ~200 face images (100 real, 100 fake) for training
4. **Jupyter Notebook** - Complete training pipeline with explanations
5. **Training Scripts** - Python scripts for model training
6. **Documentation** - Comprehensive README and instructions

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 2GB RAM minimum
- ~500MB disk space

### Installation

```bash
# Navigate to the project directory
cd streamlit_app

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Use Pre-trained Model (Fast)

The model has already been trained and is ready to use!

```bash
cd streamlit_app

# Run the web application
streamlit run app.py

# Or use the startup script
chmod +x run_app.sh
./run_app.sh
```

Open your browser and go to: `http://localhost:8501`

#### Option 2: Train from Scratch

If you want to train the model yourself:

**Using Jupyter Notebook (Recommended for Learning):**
```bash
cd streamlit_app
jupyter notebook training_notebook.ipynb
```

Then run all cells in the notebook.

**Using Python Script:**
```bash
cd streamlit_app
python train.py
```

**Then run the app:**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
streamlit_app/
├── app.py                      # Main Streamlit web application
├── model.py                    # CNN model architecture
├── train.py                    # Training script
├── training_notebook.ipynb     # Jupyter notebook for training
├── create_sample_dataset.py    # Dataset generation script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── run_app.sh                  # Startup script
│
├── dataset/                    # Training data
│   ├── real/                   # Real face images (100 images)
│   ├── fake/                   # AI-generated faces (100 images)
│   └── README.md               # Dataset documentation
│
├── models/                     # Saved model weights
│   └── face_detector_model.h5  # Trained CNN model (53MB)
│
└── .streamlit/                 # Streamlit configuration
    └── config.toml             # App settings
```

## 🎯 How to Use the Application

1. **Start the App**: Run `streamlit run app.py`
2. **Upload Image**: Click on the file uploader and select a face image (JPG, JPEG, or PNG)
3. **Analyze**: Click the "🔎 Analyze Image" button
4. **View Results**: See the prediction with confidence score:
   - ✅ REAL FACE - If the model detects a real human face
   - ⚠️ AI-GENERATED FACE - If the model detects a fake/AI-generated face

## 🧠 Model Details

### Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 128x128x3 RGB images
- **Layers**: 4 Conv blocks + 2 Dense layers
- **Output**: Binary classification (Real vs Fake)
- **Parameters**: ~2 million trainable parameters

### Performance
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~100% (on small synthetic dataset)
- **Training Time**: ~2-3 minutes (20 epochs)

### Output Interpretation
- **Score > 0.5**: Real Face (Human)
- **Score ≤ 0.5**: Fake Face (AI-generated)

## 📊 Dataset Information

The project includes a sample dataset with:
- **Real Faces**: 100 synthetic images simulating real face patterns
- **Fake Faces**: 100 synthetic images simulating AI-generated patterns
- **Total**: 200 images for demonstration

**Note**: These are synthetic images for demonstration. For a production model, use real datasets like:
- CelebA (Real faces)
- StyleGAN generated images (Fake faces)
- FFHQ dataset

### Adding More Data

To improve accuracy, add more images:

```bash
# Add real face images
cp your_real_images/* streamlit_app/dataset/real/

# Add fake/AI-generated images
cp your_fake_images/* streamlit_app/dataset/fake/

# Retrain the model
cd streamlit_app
python train.py
```

## 🔧 Customization

### Modify Training Parameters

Edit `train.py` or the notebook:

```python
epochs = 20              # Number of training epochs
batch_size = 16          # Batch size for training
img_size = (128, 128)    # Image dimensions
learning_rate = 0.001    # Learning rate
```

### Modify Model Architecture

Edit `model.py` to change the CNN structure:

```python
# Add more convolutional layers
# Adjust filter sizes
# Change dropout rates
# Modify dense layer sizes
```

### Customize UI

Edit `app.py` to modify:
- Color scheme
- Layout
- Text and messages
- Styling

## 📈 Training Process

The training process includes:

1. **Data Loading**: Loads images from dataset folders
2. **Preprocessing**: Resizes images to 128x128 and normalizes pixel values
3. **Data Split**: 80% training, 20% validation
4. **Model Training**: Trains CNN with early stopping and learning rate reduction
5. **Evaluation**: Calculates accuracy, precision, recall, and F1-score
6. **Model Saving**: Saves best model weights to `models/face_detector_model.h5`

### Training Outputs

- `models/face_detector_model.h5` - Trained model weights
- `training_history.png` - Training/validation accuracy and loss plots
- `confusion_matrix.png` - Confusion matrix visualization
- `sample_predictions.png` - Sample predictions on validation set

## 🎓 For College Demonstration

### What to Present

1. **Project Overview** (2 minutes)
   - Explain the problem: Real vs Fake face detection
   - Show the tech stack: Python, TensorFlow, Streamlit

2. **Live Demo** (5 minutes)
   - Run the Streamlit app
   - Upload sample images
   - Show predictions and confidence scores

3. **Code Walkthrough** (5 minutes)
   - Show model architecture in `model.py`
   - Explain training process in notebook
   - Demonstrate how the app works in `app.py`

4. **Results Discussion** (3 minutes)
   - Show training plots
   - Discuss accuracy and performance
   - Explain limitations and improvements

### Key Points to Mention

✅ Complete end-to-end ML pipeline  
✅ Custom CNN architecture built from scratch  
✅ User-friendly web interface  
✅ Trained on sample dataset  
✅ Binary classification with confidence scores  
✅ Fully documented code and process  

## 🛠️ Troubleshooting

### Common Issues

**1. Model Not Found Error**
```
Solution: Train the model first
cd streamlit_app
python train.py
```

**2. Import Errors**
```
Solution: Install dependencies
pip install -r requirements.txt
```

**3. Out of Memory**
```
Solution: Reduce batch size in train.py
batch_size = 8  # Instead of 16
```

**4. Dataset Not Found**
```
Solution: Run dataset generation script
python create_sample_dataset.py
```

**5. Streamlit Not Starting**
```
Solution: Check port availability
streamlit run app.py --server.port 8502
```

## 📚 Dependencies

Main libraries used:
- **streamlit** - Web application framework
- **tensorflow** - Deep learning framework
- **opencv-python** - Image processing
- **pillow** - Image handling
- **numpy** - Numerical operations
- **scikit-learn** - Data splitting and metrics
- **matplotlib** - Plotting and visualization
- **jupyter** - Interactive notebooks

See `requirements.txt` for complete list.

## ⚠️ Important Notes

1. **Educational Purpose**: This is a proof-of-concept for college demonstration
2. **Synthetic Dataset**: Uses generated images, not real face datasets
3. **Limited Accuracy**: Production models need larger, diverse datasets
4. **Not for Production**: Would need improvements for real-world use:
   - Larger dataset (10,000+ images)
   - More sophisticated architecture
   - Better preprocessing
   - Extensive testing
   - Security measures

## 🎯 Learning Outcomes

By completing this project, you demonstrate:

- ✅ Building CNN models from scratch
- ✅ Training deep learning models
- ✅ Creating web applications with Streamlit
- ✅ Image preprocessing and handling
- ✅ Model evaluation and metrics
- ✅ Complete ML pipeline implementation
- ✅ Documentation and code organization

## 📧 Support

For issues with this project:

1. Check the troubleshooting section
2. Review code comments in files
3. Read the Jupyter notebook explanations
4. Check console output for error messages

## 📄 License

This project is created for educational purposes as a college activity demonstration.

## 🎉 Success Checklist

Before your presentation, ensure:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset exists (200 images in dataset/real and dataset/fake)
- [ ] Model trained (models/face_detector_model.h5 exists)
- [ ] App runs successfully (`streamlit run app.py`)
- [ ] Can upload and analyze images
- [ ] Predictions display correctly with confidence scores
- [ ] Training notebook runs without errors
- [ ] Understand the model architecture
- [ ] Can explain the complete workflow

---

**Project Created**: November 2025  
**Purpose**: College Activity Demonstration  
**Tech Stack**: Python + TensorFlow + Streamlit  
**Status**: ✅ Complete and Ready for Demo
