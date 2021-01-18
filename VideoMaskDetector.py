# python VideoMaskDetector.py

# packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="MaskDetector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

def FindPredictMask(frame, net, maskNet):
	(h, w) = frame.shape[:2] #dimensions of the fram
	blob = cv2.dnn.blobFromImage(frame, scalefactor=1, size=(300, 300), mean=(104.0, 177.0, 123.0)) #mean subtraction vals

	# pass the blob through the network and obtain the face detections
	net.setInput(blob)
	detections = net.forward()

	faces = []
	locs = [] #1X4 matrix, box of face
	preds = [] #1X2 matrix, [mask confidence, no mask confidence]

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2] #confidence (probability) from each detection

		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int") #x,y coordinates of box

			# makes sure boxes is within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI resize it to 224x224, preprocess it, and append them
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		# batch predictions on all faces at the same time
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)



# loading serialized face detector model from disk
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_iter_150000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("loading face mask detector model...")
maskNet = load_model(args["model"])
print("starting video stream, camera is warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.5)

while True:
	cut = vs.read()
	frame = imutils.resize(cut, width=400)
	(locs, preds) = FindPredictMask(frame, net, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask >= withoutMask else "No Mask"
		print(mask)
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) #probability in the label

		cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Mask Detector", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	if cv2.waitKey(1) == ord('p'):
		while True:
			key = cv2.waitKey(1) or 0xff
			cv2.imshow('frame', frame)
			if key == ord('p'):
				break
cv2.destroyAllWindows()
vs.stop()