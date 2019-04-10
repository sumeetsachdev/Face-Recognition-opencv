import cv2
import os
import numpy as np
from PIL import Image

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        name = (os.path.split(image_path)[1].split("_")[0])
        name = int(name.replace(name[:6], ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(name)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(500)
    return images, labels

path = 'PNG' 
path1 = "Test"
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))


image_paths = [os.path.join(path1, f) for f in os.listdir(path1)]
print(image_paths)
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        name_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        name_actual = os.path.split(image_path)[1].split("_")[0]
        name_actual = int(name_actual.replace(name_actual[:6], ""))
        if name_actual == name_predicted:
            print("{} is Correctly Recognized with confidence {}".format(name_actual, conf))
        else:
            print("{} is Incorrectly Recognized as {}".format(name_actual, name_predicted))
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
