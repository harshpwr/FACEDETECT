# Sample Dataset

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
