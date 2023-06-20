import pickle
import cv2
import numpy as np
from hog import HogDescriptor


model = pickle.load(open("svm_model.pickle", "rb"))


def extract_hog_features(image):
    img = cv2.resize(image, (128, 128))
    print(img.shape)
    hog = HogDescriptor(img, cell_size=8, bin_size=9)
    vector, hog_image = hog.extract()
    vector = np.array(vector)
    vector = vector.flatten()
    return vector


cap_region_x_begin = 0
cap_region_y_end = 0.4
cam = cv2.VideoCapture(0)


def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Cac thong so lay threshold
threshold = 60
blurValue = 41
bgSubThreshold = 50  # 50
learningRate = 0


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


while True:
    ret, frame = cam.read()
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (int(frame.shape[1] * 0.4), int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("ASL Recognization", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k == ord('q'):
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        img = remove_background(frame)
        # Lay vung detection
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):int(frame.shape[1] * 0.4)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = extract_hog_features(img)
        features = np.reshape(features, (1, 8100))
        result = model.predict(features)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        __draw_label(
            img, 'SVM: '+result[0], (20, 20), (255, 0, 0))

       # Display the resulting frame
        cv2.imshow('Frame', img)

cam.release()

cv2.destroyAllWindows()
