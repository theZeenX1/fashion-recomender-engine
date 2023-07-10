import cv2 as cv
import numpy as np

faceConfig = r'files\opencv_face_detector.pbtxt'
faceModel = r'files\opencv_face_detector_uint8.pb'

ageConfig = r'models\age.prototxt'
ageModel = r'models\dex_imdb_wiki.caffemodel'

genderConfig = r'models\gender.prototxt'
genderModel = r'models\gender_net.caffemodel'

faceNet = cv.dnn.readNet(faceModel, faceConfig)
ageNet = cv.dnn.readNet(ageModel, ageConfig)
genderNet = cv.dnn.readNet(genderModel, genderConfig)

###-----Model specific mean values and categories------###
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = np.arange(101)
genderList = ['Male', 'Female']



def ageGroup(x):
    if (x <= 12):
        return "Kid"
    elif (x > 12 and x <= 18):
        return "Teen"
    elif (x > 18 and x < 25):
        return "Young Adult"
    else:
        return "Adult"


def faceBox(faceNet, frame):
    height = frame.shape[0]
    width = frame.shape[1]

    blob = cv.dnn.blobFromImage(frame, 1, (300, 300), [103.93, 116.77, 123.68], True,
                                False)  # mean values obtained from pysearchImage
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.95:
            x1 = int(detection[0, 0, i, 3] * width)
            y1 = int(detection[0, 0, i, 4] * height)
            x2 = int(detection[0, 0, i, 5] * width)
            y2 = int(detection[0, 0, i, 6] * height)
            bbox.append([x1, y1, x2, y2])
            cv.rectangle(frame, (x1, y1), (x2, y2), (75, 0, 130), int(round(height / 150)), 8)
    return frame, bbox


def openCam():
    cam = cv.VideoCapture(0, cv.CAP_DSHOW)

    width = 540
    height = 675

    cam.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv.CAP_PROP_FPS, 30)
    cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    while cam.isOpened():
        success, frame = cam.read()
        frame = cv.flip(frame, 1)
        img = cv.imread(r'C:\Users\ZeenX1\Desktop\images.jpeg')
        cv.imshow('Camera Feed', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()
    return detectCategory(img)


def detectCategory(frame):
    frame = cv.flip(frame, 1)
    bframe, bboxes = faceBox(faceNet, frame)

    padding = 20

    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding): min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding): min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv.dnn.blobFromImage(face, 1, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
        blob2 = cv.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # gender detect
        genderNet.setInput(blob2)
        genderPrediction = genderNet.forward()
        # print(genderPrediction[0])
        # t = 0
        # if(genderPrediction[0][0] > genderPrediction[0][1]):
        #     t = 0
        # else:
        #     t = 1
        gender = genderList[genderPrediction[0].argmax()]

        # age detect
        ageNet.setInput(blob)
        agePrediction = ageNet.forward()
        age = ageList[agePrediction[0].argmax()]
        age = ageGroup(age)
        prop = []
        prop.append(gender);
        prop.append(age);
        return prop

# x = openCam()
# print(x)

