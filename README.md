
# deep-cnn-images

 ## Overview

This Deep Learning application is designed to classify images into two categories: hats and sunglasses. It utilizes a Convolutional Neural Network (CNN) implemented with TensorFlow and OpenCV for image preprocessing and model training.

  **TODO:** test for
 - No hat, no sunglasses  
 - Has had *and* sunglasses
 - Script for image normalization (cropping, sizing, convert to b&w)

### Features

- Image Preprocessing: Validates and filters images based on file type and removes corrupted images.
- Image Display: Shows images within the dataset using Matplotlib.
- Data Normalization and Augmentation: Scales pixel values for neural network suitability.
- Model Training and Validation: Trains the CNN with real-time logging via TensorBoard.
- Performance Evaluation: Evaluates the model using precision, recall, and accuracy metrics.
- Prediction: Demonstrates the model's prediction capability on new images.

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- OpenCV
- Matplotlib
- NumPy

### Installation

Clone the repository:

```
git clone https://github.com/your-repository/deep-learning-app.git

cd deep-learning-app
```

Install dependencies:

```
pip install -r requirements.txt
```
### Usage

#### Prepare your dataset:

Ensure that your data is stored in the data directory with subfolders named hats and sunglasses for respective image classes.

#### Train the model:

Run the script to preprocess the data, train the model, and evaluate it:

```
python train_model.py
```

#### Evaluate the model:

After training, the model's performance metrics are displayed, and the model is tested on new images.

#### Make predictions:

Use the model to classify new images by modifying the path in the prediction section of the script.

### Model Architecture

- Input Layer: 256x256 RGB images.
- Convolutional and Pooling Layers: Three sets of convolution and max pooling for feature extraction.
- Fully Connected Layers: One dense layer with 256 units followed by a sigmoid output layer for binary classification.

### Outputs

- Training Progress: Viewable in TensorBoard to monitor loss and accuracy.
- Model Metrics: Precision, recall, and accuracy are printed after evaluating the test set.
- Predictions: The script outputs the predicted class of input images.

### Saving and Loading the Model

The trained model is saved as headwear.h5 in the models directory. It can be reloaded using TensorFlow's load_model function to make further predictions.
