from imutils import face_utils
import dlib
import cv2
import glob
import face_recognition
import itertools

from model import emotions
import numpy as np
from sklearn import svm, manifold, decomposition, discriminant_analysis
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from draw_dataset import plot_embedding


# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Emotion list
# shapePredictorPath = 'models/shape_predictor_68_face_landmarks.dat'
# faceDetector = dlib.get_frontal_face_detector()
# facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)
# print 'Landmark predictor load done'


# data = {}
# training_set_size = 0.8


facial_features = [
        'chin',
        'left_eyebrow',     # 1
        'right_eyebrow',    # 2
        'nose_bridge',      # 3
        'nose_tip',         # 4
        'left_eye',         # 5
        'right_eye',        # 6
        'top_lip',          # 7
        'bottom_lip'        # 8
    ]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          axis=1):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
        # cm = 2 * cm.astype('float') /(cm.sum(axis=0) + cm.sum(axis=1))
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_facelandmark(grayFace):
    # global faceDetector, facialLandmarkPredictor
    # face = image_as_nparray(grayImage)
    # face = faceDetector(grayImage, 1)
    # if len(face) == 0:
    #     return None

    # facialLandmarks = facialLandmarkPredictor(grayImage, face[0])
    # facialLandmarks = face_utils.shape_to_np(facialLandmarks)
    #
    # (x31, y31) = facialLandmarks[30]
    # xyList = []
    # for (x, y) in facialLandmarks[17:]:
    #     # xyList.append((x - x31, y - y31))
    #     xyList.append(x)
    #     xyList.append(y)
    #
    # # normalize
    # xyArray = np.array(xyList)
    # mu = xyArray.mean()
    # # sigma = xyArray.std()
    # # xyArray = 100.0 * (xyArray - mu)/ sigma
    # max = xyArray.max()
    # min = xyArray.min()
    # # print max, min
    # # xyArray = 100.0 * (xyArray - min) / (max - min)
    # # xyList = list(xyArray)

    face_landmarks_list = face_recognition.face_landmarks(grayFace)
    # print face_landmarks_list
    return face_landmarks_list


def get_feature(face_landmarks_list):
    eyebrow_feature, eye_feature, mouth_feature = [], [], []
    # xyPoints = [(xyList[::2], xyList[1::2])]

    # # eyebrows
    # eyebrow_feature += (xyList[8] - xyList[20], xyList[9] - xyList[21]) # point[4] - point[10] (original [22] - [28])
    # eyebrow_feature += (xyList[10] - xyList[20], xyList[11] - xyList[21])  # point[5] - point[10] (original [23] - [28])
    # eyebrow_feature += (xyList[8] - xyList[0], xyList[9] - xyList[1]) # point[4] - point[0] (original [22] - [18])
    # eyebrow_feature += (xyList[10] - xyList[18], xyList[11] - xyList[19])  # point[4] - point[9] (original [23] - [27])
    #
    # # eyes
    # eye_feature += (xyList[44] - xyList[38], xyList[45] - xyList[39]) # point[22] - point[19] (original [40] - [37])
    # eye_feature += (xyList[50] - xyList[36], xyList[51] - xyList[37]) # point[25] - point[18] (original [43] - [36])
    # eye_feature += (xyList[20] + xyList[22] - xyList[46] - xyList[48],
    #                 xyList[21] + xyList[23] - xyList[47] - xyList[49])
    #                                 # point[10]+[11]-[23]-[24] (original [38] - [42] + [39] - [41])
    # eye_feature += (xyList[52] + xyList[54] - xyList[58] - xyList[60],
    #                 xyList[53] + xyList[55] - xyList[59] - xyList[61])
    #                                 # point[26]+[27]-[29]-[30] (original [44] - [48] + [45] - [47])
    #
    # # mouth
    # mouth_feature += (xyList[26] - xyList[62], xyList[27] - xyList[63]) # point[13] - point[31] (original [31] - [49])
    # mouth_feature += (xyList[26] - xyList[74], xyList[27] - xyList[75])  # point[13] - point[37] (original [31] - [55])
    # mouth_feature += (xyList[80] - xyList[62], xyList[81] - xyList[63]) # point[40] - point[31] (original [58] - [49])
    # mouth_feature += (xyList[80] - xyList[74], xyList[81] - xyList[75])  # point[40] - point[37] (original [58] - [55])
    #
    # mouth_feature += (xyList[62] - xyList[74], xyList[63] - xyList[75]) # point[31] - point[37] (original [49] - [55])
    # mouth_feature += (xyList[64] + xyList[66] + xyList[68] + xyList[70] + xyList[72]
    #                   - xyList[84] - xyList[82] - xyList[80] - xyList[78] - xyList[76],
    #                   xyList[65] + xyList[67] + xyList[69] + xyList[71] + xyList[73]
    #                   - xyList[85] - xyList[83] - xyList[81] - xyList[79] - xyList[77])
    #                                 # point[32]+[33]+[34]+[35]+[36] - [42]-[41]-[40]-[39]-[38]
    #                                 # (original [50]+[51]+[52]+[53]+[54] - [60]-[59]-[58]-[57]-[56])


    eyebrow_feature = face_landmarks_list[0][facial_features[1]] + face_landmarks_list[0][facial_features[2]]
    eye_feature = face_landmarks_list[0][facial_features[5]] + face_landmarks_list[0][facial_features[6]]
    mouth_feature = face_landmarks_list[0][facial_features[7]] + face_landmarks_list[0][facial_features[8]]

    features = eyebrow_feature + eye_feature + mouth_feature
    xyfeatures = []
    for t in features:
        xyfeatures += list(t)
    return xyfeatures

import os

def get_data():
    data = []
    labels = []
    neutral_list = []

    files = glob.glob("dataset/neutral/*")
    neutralSize = files.__len__()
    print "load neutral pics"
    count = 1
    for file in files:
        # print file
        image = cv2.imread(file)  # open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        l = get_facelandmark(gray)
        new_neutral = get_feature(l)
        if count == 1:
            neutral = new_neutral
        else:
            for i in range(len(neutral)):
                neutral[i] = new_neutral[i] * 1.0 / count + neutral[i] * 1.0 * (count - 1) / count
        neutral_list.append(new_neutral)  # append image array to training data list
        count += 1

    new_neutral_list = []
    for new_neutral in neutral_list:
        new_neutral_list.append([new_neutral[i] - neutral[i] for i in range(len(neutral))])
    neutral_list = new_neutral_list

    # new_neutral_list = []
    # for new_neutral in neutral_list:
    #     new_neutral_list.append([x - y for x in new_neutral for y in neutral])
    # neutral_list = new_neutral_list

    print "...Neutral emotion:", neutral
    # print neutral_list
    # exit()

    for emotion in emotions:
        files = glob.glob("dataset/%s/*" % emotion)
        if emotion is "neutral": continue
        print "load %s pics" % emotion

        for file in files:
            # print file
            image = cv2.imread(file)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            l = get_facelandmark(gray)
            f = get_feature(l)
            # print len(f)
            # data.append(get_feature(l))
            data.append([f[i] - neutral[i] for i in range(len(f))])  # append image array to training data list
            # data.append([x - y for x in f for y in neutral])  # append image array to training data list
            labels.append(emotions.index(emotion))
        # print labels
    x = np.array(data)
    y = np.array(labels)
    neutral_array = np.array(neutral_list)

    print "load data END"
    print x.shape, x
    print y.shape, y
    print neutral_array.shape, neutral_array

    return x, y, neutral, neutral_array, neutralSize


def train(x_train, y_train, label=None):
    # SVM
    print "Train Model..."
    clf_linear  = svm.SVC(kernel='linear').fit(x_train, y_train)
    clf_poly    = svm.SVC(kernel='poly', degree=3).fit(x_train, y_train)
    clf_rbf     = svm.SVC().fit(x_train, y_train)
    clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x_train, y_train)

    joblib.dump(clf_linear, "models/%s_clf_linear.m" %label)
    joblib.dump(clf_poly, "models/%s_clf_poly.m" %label)
    joblib.dump(clf_rbf, "models/%s_clf_rbf.m" %label)
    joblib.dump(clf_sigmoid, "models/%s_clf_sigmoid.m" %label)

    print "models saved"


if __name__ == '__main__':
    data, labels, neutral, neutral_data, neutralSize = get_data()
    joblib.dump(neutral, "neutral")
    joblib.dump(neutral_data, "neutral_data")
    joblib.dump(data, "data")
    joblib.dump(labels, "labels")
    joblib.dump(neutralSize, "neutralSize")


    # joblib.dump(neutral, "neutral_a")
    # joblib.dump(neutral_data, "neutral_data_a")
    # joblib.dump(data, "data_a")
    # joblib.dump(labels, "labels_a")
    # joblib.dump(neutralSize, "neutralSize_a")

    # joblib.dump(neutral, "neutral_r")
    # joblib.dump(neutral_data, "neutral_data_r")
    # joblib.dump(data, "data_r")
    # joblib.dump(labels, "labels_r")
    # joblib.dump(neutralSize, "neutralSize_r")

    # neutralSize = joblib.load("neutralSize_a")
    # neutral = joblib.load("neutral_a")
    # neutral_data = joblib.load("neutral_data_a")
    # data = joblib.load("data_a")
    # labels = joblib.load("labels_a")

    # FLAG = "absolute"
    # label1 = "ab_neutral"
    # label2 = "ab"

    # neutralSize = joblib.load("neutralSize_r")
    # neutral = joblib.load("neutral_r")
    # neutral_data = joblib.load("neutral_data_r")
    # data = joblib.load("data_r")
    # labels = joblib.load("labels_r")
    #
    # FLAG = "relative-redundant"
    # label1 = "New_neutral"
    # label2 = "New"

    neutralSize = joblib.load("neutralSize")
    neutral = joblib.load("neutral")
    neutral_data = joblib.load("neutral_data")
    data = joblib.load("data")
    labels = joblib.load("labels")
    print labels

    FLAG = "relative"
    label1 = "neutral"
    label2 = "None"

    print data.shape, labels.shape, neutral_data.shape, neutralSize


    '''
    # visualize data
    feature_label = "eyebrow-eye-mouth (%s)" %FLAG
    # feature_label = "eyebrow-eye-mouth (absolute)"
    print feature_label
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    title = "t-SNE embedding - %s" % feature_label
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(data)
    plot_embedding(X_tsne, labels, title)
    plt.savefig("results/" + title)

    # PCA embedding of the digits dataset
    print("Computing PCA embedding")
    title = "PCA embedding - %s" % feature_label
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(data)
    plot_embedding(X_pca, labels, title)
    plt.savefig("results/" + title)

    # LDA embedding of the digits dataset
    print("Computing LDA embedding")
    title = "LDA embedding - %s" % feature_label
    X2 = data.copy() * 1.
    X2.flat[::data.shape[1] + 1] = 0.01  # Make X invertible
    X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, labels)
    plot_embedding(X_lda, labels, title)
    plt.savefig("results/" + title)
    '''

    '''titles = ['LinearSVC (linear kernel)',
              'SVC with polynomial (degree 3) kernel',
              'SVC with RBF kernel',
              'SVC with Sigmoid kernel']
    '''

    # SVM


    # detect neutral
    label = "neutral"
    print "\nDetect neutral... label %s implies mode..." %label1
    nlabels = [0] * neutralSize + [1] * len(labels)
    ndata = np.concatenate((neutral_data,data))

    nclf_linear = svm.SVC(kernel='linear')
    scores = cross_val_score(nclf_linear, ndata, nlabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    nclf_poly = svm.SVC(kernel='poly')
    scores = cross_val_score(nclf_poly, ndata, nlabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    nclf_rbf = svm.SVC(kernel='rbf')
    scores = cross_val_score(nclf_rbf, ndata, nlabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    nclf_sigmoid = svm.SVC(kernel='sigmoid')
    scores = cross_val_score(nclf_sigmoid, ndata, nlabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    joblib.dump(nclf_linear, "models/%s_clf_linear.m" % label1)
    joblib.dump(nclf_poly, "models/%s_clf_poly.m" % label1)
    joblib.dump(nclf_rbf, "models/%s_clf_rbf.m" % label1)
    joblib.dump(nclf_sigmoid, "models/%s_clf_sigmoid.m" % label1)



    # classify
    print "\nClassify emotion, label %s implies mode..." %label2

    clf_linear = svm.SVC(kernel='linear')
    scores = cross_val_score(clf_linear, data, labels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


    clf_poly = svm.SVC(kernel='poly')
    scores = cross_val_score(clf_poly, data, labels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    clf_rbf = svm.SVC(kernel='rbf')
    scores = cross_val_score(clf_rbf, data, labels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    clf_sigmoid = svm.SVC(kernel='sigmoid')
    scores = cross_val_score(clf_sigmoid, data, labels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


    joblib.dump(clf_linear, "models/%s_clf_linear.m" % label2)
    joblib.dump(clf_poly, "models/%s_clf_poly.m" % label2)
    joblib.dump(clf_rbf, "models/%s_clf_rbf.m" % label2)
    joblib.dump(clf_sigmoid, "models/%s_clf_sigmoid.m" % label2)

    '''
    label3 = "sub"
    sublist = [1, 3, 6]
    print "\nSub Classify... label %s implies mode..." %label3
    sublabels = [x for x in labels if x in sublist]
    subdata = np.concatenate([data[i] for i in range(len(labels)) if labels[i] in sublist]).reshape(-1, data.shape[1])
    print len(sublabels), subdata.shape

    nclf_linear = svm.SVC(kernel='linear')
    scores = cross_val_score(nclf_linear, subdata, sublabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    nclf_poly = svm.SVC(kernel='poly')
    scores = cross_val_score(nclf_poly, subdata, sublabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    nclf_rbf = svm.SVC(kernel='rbf')
    scores = cross_val_score(nclf_rbf, subdata, sublabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    nclf_sigmoid = svm.SVC(kernel='sigmoid')
    scores = cross_val_score(nclf_sigmoid, subdata, sublabels, cv=5)
    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    joblib.dump(nclf_linear, "models/%s_clf_linear.m" % label3)
    joblib.dump(nclf_poly, "models/%s_clf_poly.m" % label3)
    joblib.dump(nclf_rbf, "models/%s_clf_rbf.m" % label3)
    joblib.dump(nclf_sigmoid, "models/%s_clf_sigmoid.m" % label3)
    '''


    # draw matrix
    '''
    nx_train, nx_test, ny_train, ny_test = train_test_split(ndata, nlabels, test_size=0.2)
    print "\ndetect neutral, train set size:", len(nx_train), ", test set size:", len(nx_test)
    train(nx_train, ny_train, label1)
    nclf_linear = joblib.load("models/%s_clf_linear.m" %label1)
    nclf_poly = joblib.load("models/%s_clf_poly.m" %label1)
    nclf_rbf = joblib.load("models/%s_clf_rbf.m" %label1)
    nclf_sigmoid = joblib.load("models/%s_clf_sigmoid.m" %label1)

    for i, clf in enumerate((nclf_linear, nclf_poly, nclf_rbf, nclf_sigmoid)):
        print "\n", clf

        answer = clf.predict(nx_train)  # training score
        print "train set score:", np.mean(answer == ny_train)
        # print(answer)
        # print(y_train)

        answer = clf.predict(nx_test)  # training score
        print "test set score:", np.mean(answer == ny_test)


    # classify emotion on this dataset,
    # we already use relative data if in relative mode
    # which means, labels will not be zero
    # label = "None"
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    print "Classify emotion, train set size:", len(x_train),", test set size:", len(x_test)

    train(x_train, y_train, label2)
    clf_linear = joblib.load("models/%s_clf_linear.m" %label2)
    clf_poly = joblib.load("models/%s_clf_poly.m" %label2)
    clf_rbf = joblib.load("models/%s_clf_rbf.m" %label2)
    clf_sigmoid = joblib.load("models/%s_clf_sigmoid.m" %label2)

    for i, clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):
        print "\n", clf

        answer = clf.predict(x_train)         # training score
        print "train set score:", np.mean(answer == y_train)
        # print y_train
        # print answer
        # cnf_matrix = confusion_matrix(y_train, answer)
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=emotions[1:], normalize=True,
        #                       title='Normalized confusion matrix')


        # print(answer)
        # print(y_train)

        answer = clf.predict(x_test)         # training score
        print "test set score:", np.mean(answer == y_test)
        cnf_matrix = confusion_matrix(y_test, answer)
        print y_test
        print answer
        plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=emotions[1:], normalize=True, axis=0,
        #                       title='Normalized confusion matrix, f1')
        # plt.show()

        plot_confusion_matrix(cnf_matrix, classes=emotions[1:], normalize=True,
                              title='Normalized confusion matrix, recall')

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=emotions[1:], normalize=True, axis=0,
                              title='Normalized confusion matrix, precision')
        plt.show()
    '''


