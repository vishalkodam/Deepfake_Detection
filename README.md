# Deepfake_Detection

**Project Overview:**
This repository hosts a deepfake detection model created using the ResNet50 pretrained architecture. The primary motivation behind this project lies in addressing the growing threat of deepfake technology, which has implications for various societal aspects, including misinformation, privacy concerns, and potential harm to individuals. The primary objective is to build a model capable of accurately distinguishing between authentic and manipulated images of individuals.

**Selection of Model:**
The decision to employ the ResNet50 pretrained model was driven by thorough research, particularly referencing the paper titled "Improved Deep Learning Model for Deepfake Detection" (https://arxiv.org/pdf/2210.00361.pdf). This paper highlighted ResNet50's exceptional performance in deepfake detection tasks, offering a robust foundation for our model while minimizing the need for training from scratch.

**Development Challenges:**
Throughout the development process, we encountered several challenges, with a significant one being the time-consuming data loading process into Google Colab. The necessity to mount the data each time the notebook was reloaded made the workflow impractical. To address this challenge, we opted to build the model using Kaggle notebooks, leveraging the "140k Real and Fake Faces" dataset from Kaggle (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) for training and evaluation. This dataset provided a diverse image set to enhance the model's accuracy.

Initially, an attempt was made to train a custom CNN architecture, resulting in low training accuracy (50%). Recognizing the need for a more powerful model, we shifted to pretrained models. The adoption of the ResNet50 pretrained model led to a significant improvement in training accuracy. However, testing accuracy remained relatively low, indicating potential overfitting. Further optimization and fine-tuning are needed to address this issue.

**Techniques and Data Preprocessing:**
For image preprocessing and preparation for the CNN, TensorFlow's ImageDataGenerator was utilized. This tool automates various image preprocessing techniques, including rescaling, data augmentation, and normalization. This approach streamlined the data preparation process, reducing the manual effort required for cleaning and preprocessing the data.

**Results and Analysis:**
Training accuracy peaked at around 90% in earlier epochs but started to decline past epoch 8. The ResNet model performed well on real images, achieving an f1-score of 59%, but the f1-score for fake images was only 36%. Potential reasons for overfitting were considered, including the absence of regularization techniques or the need for additional preprocessing methods, as suggested in related research (https://openaccess.thecvf.com/content/ICCV2021W/RPRMI/papers/Das_Towards_Solving_the_DeepFake_Problem_An_Analysis_on_Improving_DeepFake_ICCVW_2021_paper.pdf).

**Code Attribution:**
In developing this deepfake detection model, we drew upon various resources and code snippets. The code for loading and preprocessing the dataset using ImageDataGenerator was adapted from the official TensorFlow documentation. Additionally, the implementation of the ResNet50 architecture was sourced from the Keras library.

We acknowledge the authors of the following resources:
- TensorFlow documentation: https://www.tensorflow.org/
- Keras library: https://keras.io/

The remaining code, including model training, evaluation, and custom modifications, was developed by us for this project.

**Future Work:**
This project serves as a foundational step for deepfake detection. Potential avenues for future improvement include:

1. Investigating other pretrained models such as VGG and XCeption to compare their performance against ResNet50.
2. Implementing regularization techniques, such as dropout or batch normalization, to address overfitting and enhance testing accuracy.
3. Exploring advanced deep learning architectures, such as attention mechanisms or adversarial training, to further enhance the model's ability to detect sophisticated deepfakes.
