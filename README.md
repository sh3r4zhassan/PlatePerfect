# Plate Perfect

## Overview
Plate Perfect is a computer vision-powered system designed to ensure consistency and quality in restaurant food presentation. By leveraging machine learning, the system evaluates plated dishes against predefined standards to help restaurants maintain visual appeal and portion accuracy. The project includes an iPad application for real-time food standardization, reducing operational costs and enhancing customer satisfaction.

## Features
- **Real-time Food Evaluation**: Uses a trained YOLOv8 model to assess the appearance of plated food. We are retraining the model on another dataset that better aligns with our specific requirements. For retraining, we fine-tune YOLOv8 using a transfer learning approach, where we initialize the model with pre-trained weights and then train it further on a curated dataset of restaurant dishes. This helps improve the accuracy of detecting ingredients, portion sizes, and plating arrangements unique to our use case.
- **Image Preprocessing**: Since YOLOv8 requires input data in a specific format, we convert image masks into polygonal annotations. Each image is processed into a text file containing polygon coordinates and corresponding labels before feeding it into the YOLOv8 training pipeline.
- **Ingredient Detection**: Identifies and classifies individual components of a dish.
- **Similarity Scoring**: Compares plated food against standard menu items to ensure consistency.
- **iPad Application**: Provides an intuitive interface for restaurant staff to capture and analyze food images.
- **Automated Feedback**: Offers immediate insights on plating quality and potential inconsistencies.


## Installation
### Prerequisites
- Python 3.8+
- TensorFlow & Keras
- OpenCV
- YOLOv8 model
- Firebase (for backend storage and real-time updates)

## Usage
1. Open the iPad application and connect an external camera.
2. Capture an image of the plated dish.
3. The system processes the image, analyzing ingredient placement and portion sizes.
4. Receive immediate feedback on whether the plating meets restaurant standards.
5. Staff can make necessary adjustments before serving.

## Model Training
- Dataset: Utilizes **UECFOODPIX**, a segmentation dataset with over 10,000 images across 101 food categories.
- Training: The YOLOv8 model is fine-tuned for food segmentation and classification.
- Optimization: Implements Mean Squared Error (MSE) loss function and ADAM optimizer for improved accuracy.

## Challenges & Future Improvements
### Challenges
- Limited availability of high-quality training data.
- Class imbalance affecting model performance.
- Deployment constraints requiring local hosting instead of AWS.

### Future Enhancements
- **REST API Integration**: Transition from Firebase listeners to API-based communication.
- **Automated Image Capture**: Detect when a dish is placed under the camera to trigger analysis.
- **Enhanced Model Performance**: Implement YOLOv8 with model pruning to balance accuracy and speed.
- **Cloud Hosting**: Deploy the model on AWS for scalability.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

