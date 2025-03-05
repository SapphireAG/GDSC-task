# **GDSC CIFAR-10 Image Classification Task**

## **Overview**  
This project involves classifying images from the **CIFAR-10 dataset**, which contains **60,000 32×32 images** across **10 different classes**. The dataset is split into:  
- **Training Set**: 50,000 images  
- **Test Set**: 10,000 images  

The goal was to train a **Convolutional Neural Network (CNN)** to achieve high accuracy on the test set. Additionally, **transfer learning** using the **VGG16 model** was explored.

---

## **Model Implementation & Performance**

### **1️. CNN Model**
- Achieved **~89% training accuracy** and **~84.25% test accuracy** in **20 epochs**.
- Utilized the following key **layers**:
  - `Conv2D` (for feature extraction)
  - `MaxPooling2D` (to reduce dimensionality)
  - `Flatten` (to convert feature maps into a vector)
  - `Dense` (fully connected layers)
  - `Dropout` (to reduce overfitting)
  - `BatchNormalization` (to speed up training and stabilize learning)
  - `Activation` (ReLU & Softmax)

- **Optimizations Implemented:**
  - **Early Stopping** (to stop training when validation loss stops improving)
  - **Learning Rate Scheduler** (`ReduceLROnPlateau` – reduces learning rate if validation loss stagnates for 3 epochs)
  - **Adam Optimizer** (for efficient gradient-based optimization)
  - **Loss & Accuracy Analysis:** No significant overfitting observed based on training curves.

---

### **2️. Transfer Learning with VGG16**
- **Used a pre-trained VGG16 model (trained on ImageNet)**
- **Modified the model:**
  - Removed the final layers
  - **Froze the convolutional layers** (to retain learned image features)
  - Added a **custom fully connected classifier**
  - Applied **Softmax activation** for probability output

- **Challenges:**
  - Due to **time, memory, and size constraints**, `model.fit()` was **not run** for transfer learning.

---

## **Technologies Used**
- **Python**
- **TensorFlow/Keras**
- **Matplotlib & Seaborn** (for visualizing training curves)
- **Kaggle** (for dataset)

---

## **Future Improvements**
- **Train the Transfer Learning Model** with fine-tuning.
- **Try Different Architectures** (e.g., ResNet, EfficientNet).
- **Data Augmentation** to improve generalization.
- **Hyperparameter Tuning** for better performance.

---

## **Conclusion**
This project successfully trained a CNN on CIFAR-10 with high accuracy. Transfer learning using VGG16 was explored but not fully trained due to constraints. The CNN model performed well without overfitting, making use of batch normalization, dropout, and learning rate scheduling.

---
