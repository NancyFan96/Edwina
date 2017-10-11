import cv2
import glob
import face_recognition

import time

resize = 350

# faceDet = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
# faceDet2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
faceDet = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
# faceDet4 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotions


def extract_faces_from_floder(flag="", path="sorted_set"):
    for emotion in emotions:
        files = glob.glob("%s/%s/*" % (path, emotion) ) # Get list of all images with emotion

        # files = glob.glob("add_smiles/*")  # Get list of all images with additional smiles

        filenumber = 0
        for f in files:
            frame = cv2.imread(f)  # Open image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

            # Detect face using 4 different classifiers
            face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

             # Cut and save face
            for (x, y, w, h) in face:  # get coordinates and size of rectangle containing face
                print "face found in file: %s" % f
                gray = gray[y:y + h, x:x + w]  # Cut the frame to size
                out = cv2.resize(gray, (resize, resize))  # Resize face so all images have same size
                cv2.imwrite("dataset/%s/%s%s.jpg" % (emotion, flag, filenumber), out)  # Write image
                print "save face"
            filenumber += 1  # Increment image number

    return


def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]

    # loop over the cat faces and draw a rectangle surrounding each
    # for (x, y, w, h) in faces_coordinates:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.imwrite("checkface/%s.jpg" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), image)


    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)


def _normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (resize, resize), interpolation=cv2.INTER_AREA)
    return face

def _locate_faces(image):
    faces = faceDet.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )
    # print "A:",faces
    # faces = face_recognition.face_locations(image)
    # print "B:", faces

    return faces  # list of (x, y, w, h)
