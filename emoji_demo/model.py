import cv2
import glob
import random
import argparse
import numpy as np
from extract_face import find_faces


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list

eigenface = cv2.face.createEigenFaceRecognizer()
fishface = cv2.face.createFisherFaceRecognizer()  # Initialize fisher face classifier
lbphface = cv2.face.createLBPHFaceRecognizer()

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


def run_test(mode = "1", check = False):
    ok, ook = train(mode)

    if ok == -1:
        return -1
    else:
        prediction_data, prediction_labels = ok, ook
        cnt = 0
        correct = 0
        incorrect = 0
        model = load_model(mode)

        for image in prediction_data:
            pred, conf = model.predict(image)
            if pred == prediction_labels[cnt]:
                if check == True:
                    cv2.imwrite("right/%s_%s_%s.jpg" % (emotions[prediction_labels[cnt]], emotions[pred], cnt),
                            image)  # <-- this one is new
                correct += 1
                cnt += 1
            else:
                if check == True:
                    cv2.imwrite("difficult/%s_%s_%s.jpg" % (emotions[prediction_labels[cnt]], emotions[pred], cnt),
                            image)  # <-- this one is new
                incorrect += 1
                cnt += 1
    return ((100 * correct) / (correct + incorrect))




def train(mode):
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    if mode == "0":
        print "training eigenface classifier, training set size: %d", len(training_labels)
        eigenface.train(training_data, np.asarray(training_labels))
        eigenface.save('models/emotion_detection_model.xml')
        print "save eigenface model as models/emotion_detection_model.xml"

        return prediction_data, prediction_labels

    elif mode == "1":
        print "training fisher face classifier, training set size:", len(training_labels)
        fishface.train(training_data, np.asarray(training_labels))
        fishface.save('models/emotion_detection_model.xml')
        print "save fisherface model as models/emotion_detection_model.xml"

        return prediction_data, prediction_labels

    elif mode == "2":
        print "training lbph face classifier, training set size:", len(training_labels)
        lbphface.train(training_data, np.asarray(training_labels))
        lbphface.save('models/emotion_detection_model.xml')
        print "save lbphface model as models/emotion_detection_model.xml"

        return prediction_data, prediction_labels

    else:
        print "wrong mode"
        return -1, -1

def load_model(mode):
    if mode == "0":
        face_model = cv2.face.createFisherFaceRecognizer()
    elif mode == "1":
        face_model = cv2.face.createFisherFaceRecognizer()
    elif mode == "2":
        face_model = cv2.face.createLBPHFaceRecognizer()
    else:
        print "wrong mode"
        return -1

    face_model.load('models/emotion_detection_model.xml')

    return face_model


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", default="1",
                    help="eigenface: 0 | fisherface: 1 | lbphface: 2")
    ap.add_argument("-t", "--time", required=True, default="1",
                    help="how many times to run test set, 0 means selecting one image using -i...")
    ap.add_argument("-c", "--check", default="f",
                    help="if check image when running test set")
    ap.add_argument("-i", "--image",
                    help="path to the test image")
    ap.add_argument("-l", "--label",
                    help="emotion label of test image")
    args = vars(ap.parse_args())

    mode = args["mode"]
    time = args["time"]
    image = args["image"]
    label = args["label"]

    # Now run it
    metascore = []

    for i in range(0, int(time)):
        correct = run_test()
        print "got", correct, "percent correct!"
        metascore.append(correct)
    if metascore.__len__() != 0:
        print "\nend score:", np.mean(metascore), "percent correct!\n"
        # 43.4->45.4 79.2->80.9 40.4->44.3

    else:
        model = load_model(mode)
        test_image = cv2.imread(image)
        for normalized_face, (x, y, w, h) in find_faces(test_image):
            prediction, conf = model.predict(normalized_face)  # do prediction
            # prediction = prediction[0]
            prediction = emotions[prediction]
            print "test image labeled as %s, predicted as %s, with conf %s" %(label, prediction, conf)




