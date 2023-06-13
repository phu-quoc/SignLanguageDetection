from sklearn.metrics import confusion_matrix
from hog import HogDescriptor
from svm import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import os
import pickle


X = []
y = []


def extract_hog_features(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    hog = HogDescriptor(img, cell_size=8, bin_size=9)
    vector, hog_image = hog.extract()
    vector = np.array(vector)
    vector = vector.flatten()
    return vector


for file in os.listdir('data'):
    features = extract_hog_features('data/' + file)
    X.append(features)
    y.append(file[0:1])
    print(file)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svm = SVM(num_classes=3, learning_rate=0.001, lambda_param=0.01, num_iterations=1000)
model = svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)

pickle.dump(model, open("svm_model.pickle", "wb"))
