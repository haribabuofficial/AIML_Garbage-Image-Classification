# AIML_Garbage-Image-Classification

Key Skills: Python, TensorFlow/Keras, OpenCV, NumPy, Scikit-learn, Transfer Learning, ResNet50, MobileNetV2, Image Augmentation, Computer Vision

* Developed a 6-class garbage waste classification system (Cardboard, Glass, Metal, Paper, Plastic, and Trash) using Computer Vision and Deep Learning to support automated waste segregation.

* Performed image preprocessing using OpenCV, including resizing, RGB conversion, normalization, and train/validation/test data splitting with Scikit-learn.
* Addressed class imbalance by generating synthetic samples for the minority Trash class using ImageDataGenerator with rotation, zoom, translation, brightness adjustment, and horizontal flipping.
Built and evaluated a custom CNN model and benchmarked it against transfer learning approaches using ResNet50 and MobileNetV2 pre-trained on ImageNet.
* Achieved 81.3% test accuracy using MobileNetV2, significantly outperforming the custom CNN (~36%) and frozen ResNet50 (~30%), demonstrating the effectiveness of lightweight transfer learning for image classification.
* Implemented an end-to-end deep learning pipeline covering data preprocessing, augmentation, model training, evaluation, and performance comparison for multi-class image classification
* Built an interactive Streamlit web application that allows users to upload garbage images and receive real-time waste category predictions.
