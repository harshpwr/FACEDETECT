# 🚀 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies (1 minute)

```bash
cd streamlit_app
pip install -r requirements.txt
```

### Step 2: Verify Setup (30 seconds)

```bash
python verify_setup.py
```

You should see: ✅ ALL CHECKS PASSED (10/10)

### Step 3: Run the App (5 seconds)

```bash
streamlit run app.py
```

Open browser: **http://localhost:8501**

---

## 📱 Using the App

1. **Upload** a face image (JPG, PNG, JPEG)
2. **Click** the "🔎 Analyze Image" button
3. **View** the prediction and confidence score

---

## 🎓 For College Demo

### Pre-Demo Checklist (2 minutes)

```bash
# 1. Verify everything works
python verify_setup.py

# 2. Start the app
streamlit run app.py

# 3. Test with a sample image from dataset/real/ or dataset/fake/
```

### Demo Flow (10 minutes)

1. **Introduction** (1 min)
   - "AI system to detect real vs fake faces"
   
2. **Live Demo** (3 min)
   - Upload real face → Show result
   - Upload fake face → Show result
   
3. **Technical Explanation** (3 min)
   - Show model.py (CNN architecture)
   - Show training results (accuracy: 100%)
   
4. **Training Process** (2 min)
   - Open training_notebook.ipynb
   - Show training plots
   
5. **Q&A** (1 min)

---

## 📊 Key Numbers to Mention

- **200 images** in training dataset
- **100% validation accuracy**
- **53 MB** model size
- **2 minutes** training time
- **<1 second** inference time
- **~2 million** parameters

---

## 🔧 Troubleshooting

### App won't start?
```bash
# Check port 8501 is free
lsof -i :8501
# Or use different port
streamlit run app.py --server.port 8502
```

### Import errors?
```bash
pip install -r requirements.txt --upgrade
```

### Model not found?
```bash
# Train the model
python train.py
# Or use the notebook
jupyter notebook training_notebook.ipynb
```

---

## 📂 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |
| `model.py` | CNN architecture definition |
| `train.py` | Model training script |
| `training_notebook.ipynb` | Interactive training notebook |
| `dataset/` | Training images (real + fake) |
| `models/` | Saved model weights |

---

## 💡 Quick Commands

```bash
# Verify setup
python verify_setup.py

# Run app
streamlit run app.py

# Train model
python train.py

# Open notebook
jupyter notebook training_notebook.ipynb

# Create new dataset
python create_sample_dataset.py
```

---

## ✅ Success Indicators

You're ready when you see:
- ✅ App loads at http://localhost:8501
- ✅ Can upload and analyze images
- ✅ Predictions show with confidence scores
- ✅ Model file exists: models/face_detector_model.h5

---

## 🆘 Need Help?

1. Check **COLLEGE_DEMO_GUIDE.md** for detailed presentation guide
2. Read **README.md** for complete documentation
3. Review code comments in Python files
4. Check troubleshooting section above

---

**Ready? Let's go! Run:** `streamlit run app.py`

---

**Project Status**: ✅ Complete & Ready  
**Last Updated**: November 2025
