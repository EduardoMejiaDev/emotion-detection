# Emotion detection face expression on real time

## Introduction

This code implements a deep learning model for emotion detection. The model is trained on a dataset of facial images labeled with seven emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised. The trained model can be used to classify emotions in real-time using a webcam.

The code uses the Keras library with a TensorFlow backend to build and train the deep learning model. It leverages convolutional neural networks (CNNs) to extract features from facial images and predict the corresponding emotion.

The trained model is saved in two files: `emotion_model.json` (containing the model architecture) and `emotion_model.h5` (containing the model weights).

## Code Explanations

### Filename: EvaluateEmotionDetector.py

- Save the model structure in a JSON file (`emotion_model.json`) and save the trained model weights in an HDF5 file (`emotion_model.h5`).
- Import the required packages and libraries.
- Define a dictionary `emotion_dict` to map emotion labels to their corresponding indices.
- Load the model architecture from the JSON file (`emotion_model.json`).
- Load the model weights from the HDF5 file (`emotion_model.h5`).
- Initialize an image data generator for testing with pixel rescaling.
- Preprocess the training and validation images using the data generators. The images are resized to (48, 48) pixels, converted to grayscale, and categorized into emotion classes.
- Perform emotion predictions on the test data using the loaded model.
- Calculate and display the confusion matrix using `sklearn.metrics.confusion_matrix`.
- Display the confusion matrix using `sklearn.metrics.ConfusionMatrixDisplay`.
- Print the classification report using `sklearn.metrics.classification_report`.

### Filename: TestEmotionDetector.py

- Import the required packages and libraries.
- Define a dictionary `emotion_dict` to map emotion labels to their corresponding indices.
- Load the model architecture from the JSON file (`emotion_model.json`).
- Load the model weights from the HDF5 file (`emotion_model.h5`).
- Start the webcam feed using `cv2.VideoCapture`.
- Define a function `generate_frames` to process the webcam frames and detect emotions.
  - Read a frame from the webcam feed.
  - Convert the frame to grayscale.
  - Detect faces in the grayscale frame using the Haar cascade classifier.
  - For each detected face:
    - Draw a bounding box around the face.
    - Preprocess the face image by resizing, converting to grayscale, and expanding dimensions.
    - Perform emotion prediction on the preprocessed face image using the loaded model.
    - Get the predicted emotion label and display it on the frame.
  - Convert the frame to JPEG format and yield it as a response.
- Define routes for the Flask application: `'/'` for rendering the HTML template and `'/video_feed'` for the video feed.
- Run the Flask application.

### Filename: TrainEmotionDetector.py

- Import the required packages and libraries.
- Initialize image data generators for training and validation with pixel rescaling.
- Preprocess the training and validation images using the data generators.
- Create the model structure using a Sequential model from Keras.
- Compile the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
- Train the model using the training data generator and validate it using the validation data generator.
- Save the model architecture as JSON in `emotion_model.json`.
- Save the trained model weights in `emotion_model.h5`.

## Usage Instructions

1. Ensure that you have a webcam connected to your computer.

2. Prepare a dataset of facial images labeled with the corresponding emotions. Organize the dataset into two folders: `data/train` and `data/test`. Place the training images in the `data/train` folder and the validation/test images in the `data/test` folder.

3. Run this code to train the emotion detection model on the provided dataset. The model will be saved in the current directory as `emotion_model.json` (model structure) and `emotion_model.h5` (model weights).

4. To use the trained model for real-time emotion detection, run the Flask application by executing the command: `python app.py`

5. Open a web browser and go to http://localhost:5000. You should see the live video feed from your webcam with emotions detected in real-time.

## Additional Information

- The dataset used for training and validation should consist of facial images labeled with emotions. Each image should be in grayscale format and of size (48, 48) pixels.

- The model architecture and hyperparameters can be adjusted to improve performance. You can experiment with different network architectures, activation functions, optimization algorithms, and learning rates.

- It's important to ensure that the training and validation datasets are representative of the real-world scenarios in which the model will be deployed. The more diverse and balanced the dataset, the better the model's performance is likely to be.

![emotion_detection](https://github.com/datamagic2020/Emotion_detection_with_CNN/blob/main/emoition_detection.png)

### Packages need to be installed

- `pip install numpy`
- `pip install opencv-python`
- `pip install keras`
- `pip3 install --upgrade tensorflow`
- `pip install pillow`
- `pip install flask`

### download FER2013 dataset

- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector

- with all face expression images in the FER2013 Dataset
- command --> `python TranEmotionDetector.py`

It will take several hours depends on your processor. (On i7 processor with 16 GB RAM it took me around 4 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file

`python TestEmotionDetector.py` at least for locally, it will run on `localhost:5000` and you must to hit `/video_feed` endpoint.
