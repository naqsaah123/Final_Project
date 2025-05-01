# Final_Project
 Alzheimer’s_Disease_via_Brain_MRI_using_Deep_Learning
 
# 🧠 Dementia Classification from MRI Images using CNN and Transfer Learning

This project implements a deep learning pipeline to classify MRI brain images into four dementia-related categories: **Mild Dementia**, **Moderate Dementia**, **Non-Demented**, and **Very Mild Dementia**. It uses a custom CNN, a deeper CNN with regularization and callbacks, and a fine-tuned **VGG16** model via transfer learning.

## 📁 Dataset Overview

- Original dataset sourced from Google Drive.
- Four categories:
  - Mild Dementia
  - Moderate Dementia
  - Non Demented
  - Very Mild Dementia
- Images are balanced via augmentation to ensure 5000 samples per class.

## 🧪 Workflow

### 1. Preprocessing
- Organized images into subfolders by class.
- Applied augmentation (rotation, zoom, horizontal flip) using `ImageDataGenerator`.
- Balanced dataset with 5000 images per class.

### 2. Data Splitting
- Split into:
  - **70% training**
  - **15% validation**
  - **15% testing**

### 3. Model Architectures

#### ✅ Baseline Custom CNN
- 3 Conv2D layers with MaxPooling
- Flatten + Dense + Dropout
- Achieved ~99.6% test accuracy

#### ⚙️ Deeper CNN with Regularization
- Batch Normalization, L2 Regularization, Dropout
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Achieved ~97% test accuracy

#### 🔁 VGG16 Transfer Learning
- Used `imagenet` weights, froze initial layers
- Unfroze last few Conv2D layers
- Added custom classifier head
- Achieved **~100% test accuracy** and perfect classification reports

## 📊 Evaluation

- Metrics: Accuracy, Precision, Recall, F1-Score
- Visualizations: Confusion Matrices and Loss/Accuracy Curves
- Tools used: `matplotlib`, `seaborn`, `scikit-learn`

### Example Classification Report (VGG16):
![image](https://github.com/user-attachments/assets/f5fefb1a-9762-4c5c-92d6-62411a91703a)


## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Google Colab + Google Drive
- Matplotlib, Seaborn, Scikit-learn
## 📈 Future Improvements

- Try other pre-trained models (e.g., ResNet50, InceptionV3)
- Apply Grad-CAM for model interpretability
- Convert to TensorFlow Lite for mobile deployment

## 👤 Author

**Naqsaah123**  
[GitHub Repo](https://github.com/naqsaah123/Final_Project)

---

