import cv2
import glob
import random
import argparse
import numpy as np
from extract_face import find_faces


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list

# eigenface = cv2.face.createEigenFaceRecognizer()
# fishface = cv2.face.createFisherFaceRecognizer()  # Initialize fisher face classifier
# lbphface = cv2.face.createLBPHFaceRecognizer()

data = {}
training_set_size = 0.8


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]  # get first 80% of file list
    prediction = files[-int(len(files) * (1 - training_set_size)):]  # get last 20% of file list
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

