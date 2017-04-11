"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
from cv2 import WINDOW_NORMAL

import cv2
from extract_face import find_faces
from image_commons import nparray_as_image, draw_with_alpha


def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    return [nparray_as_image(cv2.imread('emoji/%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show_webcam_and_run(model, emoticons, window_size=None, window_name='webcam', update_time=10):
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

    while read_value:
        i = 1
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            prediction = model.predict(normalized_face)  # do prediction
            prediction = prediction[0]
            # prediction = emotions_map[prediction]
            image_to_draw = emoticons[prediction]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, w, h))
            cv2.putText(webcam_image, (emotions[prediction]+" #{}").format(i), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            i += 1

        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    emotions =  ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    # emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    # emotions_map = {0:0, 1:1, 2:0, 3:2, 4:0, 5:3, 6:4, 7:5 }
    emoticons = _load_emoticons(emotions)
    

    fisher_face = cv2.face.createFisherFaceRecognizer()
    fisher_face.load('models/emotion_detection_model.xml')

     # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(fisher_face, emoticons, window_size=(1600, 1200), window_name=window_name, update_time=8)
