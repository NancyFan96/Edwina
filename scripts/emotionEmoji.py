"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
from cv2 import WINDOW_NORMAL
import cv2
import time
from extract_face import find_faces, resize
from image_commons import draw_with_alpha
from model import emotions
from landmark_face import get_facelandmark, get_feature, facial_features
from draw_dataset import _load_emoticons

from sklearn.externals import joblib
import numpy as np

emoticons = _load_emoticons(emotions)
print "[INFO] Load Emotion Icons"

clf_linear = joblib.load("models/clf_linear.m")
clf_poly = joblib.load("models/clf_poly.m")
clf_rbf = joblib.load("models/clf_rbf.m")
clf_sigmoid = joblib.load("models/clf_sigmoid.m")
model = clf_poly
print "[INFO] Load Classify Model"

def which_emotion(featureList):
    featureList = get_feature(featureList)
    featureArray = np.array(featureList).reshape(1, -1)
    pred = model.predict(featureArray)[0]  # do prediction

    return pred, featureList


def draw_face_landmark(featureList, (x, y, w, h), img):
    Xs = featureList[::2]
    Ys = featureList[1::2]
    x_nose = Xs[31 - 18] * w / resize
    y_nose = Ys[31 - 18] * h / resize
    cv2.circle(img, (x, y), 5, (255, 0, 0), 0)
    cv2.circle(img, (x + x_nose, y + y_nose), 5, (255, 0, 0), 0)

    for i in range(len(Xs)):
        xx = Xs[i] * w / resize
        yy = Ys[i] * h / resize
        cv2.circle(img, (x + xx, y + yy), 3, (0, 0, 255), 0)
    return img


def draw_emoji(pred, (x, y, w, h), img):
    image_to_draw = emoticons[pred]
    draw_with_alpha(img, image_to_draw, (x, y, w, h))
    cv2.putText(img, (emotions[pred] + " #{}").format(pred), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return img


def show_webcam_and_run(model, neutral_filter, neutral_face, emoticons, window_size=None, window_name='emotionEmoji', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param emoticons: List of emotions images.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return

    count = 0
    total = 0
    count_notopen = 0
    while read_value:
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            if normalized_face is None: continue
            face_landmarks_list = get_facelandmark(normalized_face)
            if face_landmarks_list is None: continue

            cv2.imwrite("checkface/normalization/original_image_%s.jpg"
                        % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), webcam_image)
            cv2.imwrite("checkface/normalization/normalized_face_%s.jpg"
                        % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), normalized_face)

            # Mark landmarks
            featureList = []
            for featureKey in facial_features:
                points = face_landmarks_list[0][featureKey]
                for point in points:
                    featureList += list(point)
            Xs = featureList[::2]
            Ys = featureList[1::2]
            for i in range(len(Xs)):
                xx = Xs[i] * w / resize
                yy = Ys[i] * h / resize
                cv2.circle(webcam_image, (x + xx , y + yy), 3, (0, 0, 255), 0)



            # predict
            total += 1
            featureList = get_feature(face_landmarks_list)
            featureArray = np.array(featureList).reshape(1, -1)
            newfeatureList = [featureList[i] - neutral_face[i] for i in range(len(featureList))]
            newfeatureArray = np.array(newfeatureList).reshape(1, -1)

            # check if it is neutral
            # check if open the mouth
            mouthfeature = featureList[44:-1:1]
            if abs(mouthfeature[21] - mouthfeature[9]) <= 1.5 * abs(mouthfeature[9] - mouthfeature[3]) + abs(mouthfeature[21] - mouthfeature[15]):
                count_notopen += 1
                neutral_check = neutral_filter.predict(newfeatureArray)[0]
                if neutral_check == 0:
                    # is neutral face
                    count += 1
                    for i in range(len(neutral_face)):
                        neutral_face[i] = featureList[i] * 1.0 / count + neutral_face[i] * 1.0 * count / (count+1)
                    pred = 0
                else:
                    newfeatureList = [featureList[i] - neutral_face[i] for i in range(len(featureList))]
                    newfeatureArray = np.array(newfeatureList).reshape(1, -1)
                    pred = model.predict(newfeatureArray)[0]  # do prediction

            else:
                newfeatureList = [featureList[i] - neutral_face[i] for i in range(len(featureList))]
                newfeatureArray = np.array(newfeatureList).reshape(1, -1)
                pred = model.predict(newfeatureArray)[0]                       # do prediction

            image_to_draw = emoticons[pred]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, 60, 60))
            cv2.putText(webcam_image, (emotions[pred] + " #{}").format(pred), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imwrite("checkface/%s.jpg" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), webcam_image)
        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            print "total: %d, neutral: %d, fraction:%f" %(total, count, count * 1.0 / total)
            print "notopen: %d, neutral: %d, fraction:%f" %(count_notopen, count, count * 1.0 / count_notopen)
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    # emotions = [("neutral"), "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list

    # SVM
    # # clf_poly = joblib.load("models/clf_poly.m")
    # clf_rbf = joblib.load("models/clf_rbf.m")
    # clf_sigmoid = joblib.load("models/clf_sigmoid.m")
    clf_linear = joblib.load("models/clf_linear.m")
    nclf_linear = joblib.load("models/neutral_clf_linear.m")

    neutral_face = joblib.load("neutral")
    neutralSize = joblib.load("neutralSize")
    neutral = joblib.load("neutral")
    neutral_data = joblib.load("neutral_data")
    data = joblib.load("data")
    labels = joblib.load("labels")

    nlabels = [0] * neutralSize + [1] * len(labels)
    ndata = np.concatenate((neutral_data,data))
    print "[INFO] Train Model..."
    clf_linear.fit(data, labels)
    nclf_linear.fit(ndata, nlabels)

    joblib.dump(clf_linear, "models/fitted_clf_linear.m")
    joblib.dump(nclf_linear, "models/fitted_nclf_linear.m")

    clf_linear = joblib.load("models/fitted_clf_linear.m")
    nclf_linear = joblib.load("models/fitted_nclf_linear.m")


    # use learnt model
    print "[INFO] Start!"
    window_name = 'EmotionEmoji (press ESC to exit)'
    show_webcam_and_run(clf_linear, nclf_linear, neutral_face, emoticons,
                        window_size=(1600, 1200), window_name=window_name, update_time=8)

