import cv2
import glob

faceDet = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
# faceDet2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
# faceDet3 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
# faceDet4 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotions


def detect_faces(emotion):
    files = glob.glob("sorted_set/%s/*" % emotion)  # Get list of all images with emotion

    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        find_faces(frame)


        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        #
        # # Detect face using 4 different classifiers
        # face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
        #                                 flags=cv2.CASCADE_SCALE_IMAGE)
        # face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
        #                                   flags=cv2.CASCADE_SCALE_IMAGE)
        # face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
        #                                   flags=cv2.CASCADE_SCALE_IMAGE)
        # face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
        #                                   flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
        # if len(face) == 1:
        #     facefeatures = face
        # elif len(face2) == 1:
        #     facefeatures == face2
        # elif len(face3) == 1:
        #     facefeatures = face3
        # elif len(face4) == 1:
        #     facefeatures = face4
        # else:
        #     facefeatures = ""

        # Cut and save face
        # for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
        #     print "face found in file: %s" % f
        #     gray = gray[y:y + h, x:x + w]  # Cut the frame to size
        #
        #     try:
        #         out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
        #         cv2.imwrite("dataset/%s/%s.jpg" % (emotion, filenumber), out)  # Write image
        #     except:
        #         pass  # If error, pass file
        # filenumber += 1  # Increment image number

def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)

def _normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face;

def _locate_faces(image):
    faces = faceDet.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces  # list of (x, y, w, h)

for emotion in emotions:
    detect_faces(emotion)  # Call functiona