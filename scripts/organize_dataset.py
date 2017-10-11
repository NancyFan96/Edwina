import glob
import argparse
import cv2
from shutil import copyfile
from extract_face import extract_faces_from_floder, find_faces


def process_ck_database():
    print "process ck database..."
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
    participants = glob.glob("source_emotion/*")  # Returns a list of all folders with participant numbers

    print participants

    for x in participants:
        part = "%s" % x[-4:]  # store current participant number
        for sessions in glob.glob("%s/*" % x):  # Store list of sessions for current participant
            for files in glob.glob("%s/*" % sessions):
                current_session = files[20:-30]
                file = open(files, 'r')

                emotion = int(
                    float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

                sourcefile_emotion = glob.glob("source_images/%s/%s/*" % (part, current_session))[
                    -1]  # get path for last image in sequence, which contains the emotion
                sourcefile_neutral = glob.glob("source_images/%s/%s/*" % (part, current_session))[
                    0]  # do same for neutral image
                print sourcefile_emotion
                print sourcefile_neutral

                dest_neut = "sorted_set/neutral/%s" % sourcefile_neutral[25:]  # Generate path to put neutral image
                dest_emot = "sorted_set/%s/%s" % (
                emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image

                copyfile(sourcefile_neutral, dest_neut)  # Copy file
                copyfile(sourcefile_emotion, dest_emot)  # Copy file

    extract_faces_from_floder()

    return


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--floder",
	help="path to the input images, the subfloder should classified by emotion")
ap.add_argument("-C", "--ck",
	help="ck database: ck")
ap.add_argument("-i", "--image",
	help="path to the input image")
ap.add_argument("-l", "--label",
	help="emotion label of input image")
ap.add_argument("-a", "--annotation", required = True,
	help="annotation for input image, used for naming")
args = vars(ap.parse_args())

floder = args["floder"]
ck = args["ck"]
image = args["image"]
image_name = image.split("/")[-1]
label = args["label"]
annotation = args["annotation"]

if ck == "ck":
    process_ck_database()
elif floder == None:
    print "only one image..."
    # input a certain image
    raw_image = cv2.imread(image)
    for normalized_face, (x, y, w, h) in find_faces(raw_image):
        print "face found in file: %s, with label %s" % (image, label)
        cv2.imwrite("dataset/%s/%s_%s.jpg" % (label, image_name, annotation), normalized_face)  # Write image
        print "save face as dataset/%s/%s_%s.jpg" %(label, image_name, annotation)
else:
    print "process all the image in floder %s" % floder
    extract_faces_from_floder(annotation, floder)



