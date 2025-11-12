"""
Script to create a sample dataset of face images for training.
This creates synthetic face-like images for demonstration purposes.

For a real project, you would:
1. Use actual face images from public datasets (e.g., CelebA, LFW)
2. Download AI-generated faces from sources like ThisPersonDoesNotExist.com
3. Use proper face detection and cropping

This script creates simple synthetic images for demonstration only.
"""

import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont

def create_real_face_sample(size=(128, 128), index=0):
    """
    Create a synthetic 'real' face image with structured patterns
    """
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Create a face-like structure with specific patterns
    # Skin tone background
    skin_tone = (220 - index % 50, 180 - index % 30, 150 - index % 40)
    img[:, :] = skin_tone
    
    # Face oval
    center = (size[0] // 2, size[1] // 2)
    cv2.ellipse(img, center, (size[0]//3, size[1]//2 + 10), 0, 0, 360, 
                (int(skin_tone[0]*0.9), int(skin_tone[1]*0.9), int(skin_tone[2]*0.9)), -1)
    
    # Eyes
    eye_y = size[1] // 3
    eye_spacing = size[0] // 4
    cv2.circle(img, (center[0] - eye_spacing, eye_y), 8, (50, 40, 30), -1)
    cv2.circle(img, (center[0] + eye_spacing, eye_y), 8, (50, 40, 30), -1)
    cv2.circle(img, (center[0] - eye_spacing, eye_y), 4, (255, 255, 255), -1)
    cv2.circle(img, (center[0] + eye_spacing, eye_y), 4, (255, 255, 255), -1)
    
    # Nose
    nose_pts = np.array([
        [center[0], eye_y + 20],
        [center[0] - 5, eye_y + 35],
        [center[0] + 5, eye_y + 35]
    ], np.int32)
    cv2.polylines(img, [nose_pts], True, (180, 140, 120), 2)
    
    # Mouth
    mouth_y = eye_y + 50
    cv2.ellipse(img, (center[0], mouth_y), (20, 10), 0, 0, 180, (150, 80, 80), 2)
    
    # Add some natural texture/noise
    noise = np.random.randint(-15, 15, (size[0], size[1], 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add slight blur for smoothness
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def create_fake_face_sample(size=(128, 128), index=0):
    """
    Create a synthetic 'fake' face image with different patterns
    """
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Different color scheme for fake faces (more artificial looking)
    base_color = (200 + index % 30, 190 + index % 25, 200 + index % 20)
    img[:, :] = base_color
    
    # More geometric/artificial face shape
    center = (size[0] // 2, size[1] // 2)
    cv2.circle(img, center, size[0]//3, 
               (int(base_color[0]*0.85), int(base_color[1]*0.85), int(base_color[2]*0.85)), -1)
    
    # Eyes - more artificial/symmetric
    eye_y = size[1] // 3
    eye_spacing = size[0] // 4
    cv2.circle(img, (center[0] - eye_spacing, eye_y), 10, (30, 30, 30), -1)
    cv2.circle(img, (center[0] + eye_spacing, eye_y), 10, (30, 30, 30), -1)
    cv2.circle(img, (center[0] - eye_spacing, eye_y), 5, (100, 200, 255), -1)
    cv2.circle(img, (center[0] + eye_spacing, eye_y), 5, (100, 200, 255), -1)
    
    # More geometric nose
    cv2.line(img, (center[0], eye_y + 15), (center[0], eye_y + 35), (160, 160, 160), 2)
    
    # Mouth - more artificial
    mouth_y = eye_y + 50
    cv2.line(img, (center[0] - 20, mouth_y), (center[0] + 20, mouth_y), (200, 100, 100), 3)
    
    # Add digital-looking patterns
    for i in range(5):
        y = size[1] // 5 * i
        cv2.line(img, (0, y), (size[0], y), (base_color[0] - 10, base_color[1] - 10, base_color[2] - 10), 1)
    
    # Add digital noise
    noise = np.random.randint(-20, 20, (size[0], size[1], 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Less smoothing for more artificial look
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def create_dataset(num_real=100, num_fake=100):
    """
    Create sample dataset with synthetic images
    """
    print("Creating sample dataset...")
    print(f"Generating {num_real} real face samples...")
    
    # Create real face images
    for i in range(num_real):
        img = create_real_face_sample(index=i)
        filename = f"real_face_{i+1:03d}.jpg"
        filepath = os.path.join('dataset', 'real', filename)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if (i + 1) % 20 == 0:
            print(f"  Created {i+1}/{num_real} real faces")
    
    print(f"\nGenerating {num_fake} fake face samples...")
    
    # Create fake face images
    for i in range(num_fake):
        img = create_fake_face_sample(index=i)
        filename = f"fake_face_{i+1:03d}.jpg"
        filepath = os.path.join('dataset', 'fake', filename)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if (i + 1) % 20 == 0:
            print(f"  Created {i+1}/{num_fake} fake faces")
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Real faces: {num_real} images in dataset/real/")
    print(f"   Fake faces: {num_fake} images in dataset/fake/")
    print(f"   Total: {num_real + num_fake} images")

def create_readme():
    """
    Create a README for the dataset folder
    """
    readme_content = """# Sample Dataset

This folder contains a sample dataset for training the face detection model.

## Structure

- `real/`: Contains real face images (~100 images)
- `fake/`: Contains AI-generated/fake face images (~100 images)

## Note

These are synthetic images created for demonstration purposes. For a production model, you would need:

1. **Real Faces**: Use datasets like:
   - CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   - LFW (Labeled Faces in the Wild): http://vis-www.cs.umass.edu/lfw/
   - FFHQ: https://github.com/NVlabs/ffhq-dataset

2. **Fake Faces**: Collect from:
   - StyleGAN generated images
   - ThisPersonDoesNotExist.com
   - Various GAN-generated face datasets

## Adding Your Own Images

To add your own images:

1. Place real face images in the `real/` folder
2. Place fake/AI-generated images in the `fake/` folder
3. Supported formats: .jpg, .jpeg, .png
4. Images will be automatically resized during training

## Dataset Size

Current dataset: ~200 images (100 real + 100 fake)

For better accuracy, consider:
- 1,000+ images per class (minimum)
- 10,000+ images per class (recommended)
- Balanced classes (equal real and fake)
- Diverse faces (age, gender, ethnicity, lighting, angles)
"""
    
    with open('dataset/README.md', 'w') as f:
        f.write(readme_content)
    
    print("\n📝 Dataset README created")

if __name__ == "__main__":
    print("="*60)
    print("Face Detection Dataset Generator")
    print("="*60)
    print("\nThis script creates a sample dataset with synthetic images.")
    print("For college demonstration purposes only.\n")
    
    # Create directories if they don't exist
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    
    # Generate dataset
    create_dataset(num_real=100, num_fake=100)
    
    # Create dataset README
    create_readme()
    
    print("\n" + "="*60)
    print("Dataset is ready! You can now:")
    print("1. Run training: jupyter notebook training_notebook.ipynb")
    print("2. Or run: python train.py")
    print("="*60)
