# import packages
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
				help="path to input image")
ap.add_argument("-f", "--face", type=str,
				default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
				default="MaskDetector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float,
				default=0.5, help="minimum probability")
args = vars(ap.parse_args())

print("loading face detector model...")
net = cv2.dnn.readNet(os.path.sep.join([args["face"], "deploy.prototxt"]), os.path.sep.join([args["face"], "res10_300x300_iter_150000.caffemodel"]))
model = load_model(args["model"])

newImg = cv2.imread(args["image"])
orig = newIMG.copy()
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
print("computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	conf = detections[0, 0, i, 2]
	if conf > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		face = cv2.resize(image[startY:endY, startX:endX], (244,244))
		face = preprocess_input(img_to_array(face))
		face = np.expand_dims(face, axis=0)
		(mask, withoutMask) = model.predict(face)[0]

		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


		cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
