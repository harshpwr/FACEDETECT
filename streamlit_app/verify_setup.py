"""
Setup Verification Script
Checks if all components are properly configured
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_directory(dirpath, description, min_files=0):
    """Check if a directory exists and has files"""
    if os.path.exists(dirpath):
        files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        count = len(files)
        status = "✅" if count >= min_files else "⚠️"
        print(f"{status} {description}: {dirpath} ({count} files)")
        return count >= min_files
    else:
        print(f"❌ {description}: {dirpath} NOT FOUND")
        return False

def check_imports():
    """Check if required libraries can be imported"""
    print("\n🔍 Checking Python Libraries:")
    libraries = [
        ('streamlit', 'Streamlit'),
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_ok = True
    for module, name in libraries:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("AI Face Detection - Setup Verification")
    print("="*60)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    checks = []
    
    print("\n📁 Checking Project Files:")
    checks.append(check_file('app.py', 'Streamlit App'))
    checks.append(check_file('model.py', 'Model Architecture'))
    checks.append(check_file('train.py', 'Training Script'))
    checks.append(check_file('training_notebook.ipynb', 'Training Notebook'))
    checks.append(check_file('requirements.txt', 'Requirements'))
    checks.append(check_file('README.md', 'Documentation'))
    
    print("\n📂 Checking Directories:")
    checks.append(check_directory('dataset/real', 'Real Face Images', min_files=10))
    checks.append(check_directory('dataset/fake', 'Fake Face Images', min_files=10))
    
    print("\n🤖 Checking Model:")
    model_exists = check_file('models/face_detector_model.h5', 'Trained Model')
    checks.append(model_exists)
    
    if not model_exists:
        print("\n⚠️  Model not found! You need to train it first:")
        print("   Option 1: jupyter notebook training_notebook.ipynb")
        print("   Option 2: python train.py")
    
    # Check libraries
    libs_ok = check_imports()
    checks.append(libs_ok)
    
    if not libs_ok:
        print("\n⚠️  Missing libraries! Install them with:")
        print("   pip install -r requirements.txt")
    
    # Summary
    print("\n" + "="*60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\n🚀 You're ready to go!")
        print("\nNext steps:")
        print("  1. Run the app: streamlit run app.py")
        print("  2. Or use: ./run_app.sh")
        print("  3. Open browser: http://localhost:8501")
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total} passed)")
        print("\nPlease fix the issues above before running the app.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
