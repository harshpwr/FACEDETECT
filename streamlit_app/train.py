import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from model import create_cnn_model

def load_dataset(dataset_path='dataset', img_size=(128, 128)):
    """
    Load images from dataset folder
    
    Args:
        dataset_path: Path to dataset folder containing 'real' and 'fake' subfolders
        img_size: Target size for images
    
    Returns:
        X: numpy array of images
        y: numpy array of labels (1 for real, 0 for fake)
    """
    images = []
    labels = []
    
    # Load real faces (label = 1)
    real_path = os.path.join(dataset_path, 'real')
    if os.path.exists(real_path):
        print(f"Loading real faces from {real_path}...")
        for img_name in os.listdir(real_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(real_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(1)  # Real = 1
        print(f"Loaded {len([l for l in labels if l == 1])} real faces")
    
    # Load fake faces (label = 0)
    fake_path = os.path.join(dataset_path, 'fake')
    if os.path.exists(fake_path):
        print(f"Loading fake faces from {fake_path}...")
        for img_name in os.listdir(fake_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fake_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(0)  # Fake = 0
        print(f"Loaded {len([l for l in labels if l == 0])} fake faces")
    
    # Convert to numpy arrays
    X = np.array(images, dtype='float32') / 255.0  # Normalize
    y = np.array(labels, dtype='float32')
    
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print(f"Real faces: {np.sum(y == 1)}")
    print(f"Fake faces: {np.sum(y == 0)}")
    
    return X, y

def train_model(epochs=20, batch_size=16):
    """
    Train the face detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Error: No images found in dataset!")
        return None, None
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Create model
    print("\nCreating model...")
    model = create_cnn_model()
    
    # Data augmentation
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation Precision: {val_precision*100:.2f}%")
    print(f"Validation Recall: {val_recall*100:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/face_detector_model.h5'
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, history

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.show()

if __name__ == "__main__":
    print("="*50)
    print("Face Detection Model Training")
    print("="*50)
    
    # Train model
    model, history = train_model(epochs=20, batch_size=16)
    
    if history is not None:
        # Plot training history
        plot_training_history(history)
        print("\n✅ Training completed successfully!")
        print("You can now run the Streamlit app: streamlit run app.py")
    else:
        print("\n❌ Training failed. Please check the dataset.")
