import face_recognition
import imutils
import pickle
import time
import cv2
import os
import urllib.request
import numpy as np
import RPi.GPIO as GPIO 

cascPathface = os.path.dirname(
cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())

PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN, GPIO.OUT)


url = "http://192.168.155.226:8080/shot.jpg"

t= 0

#print("Streaming started")
#video_capture = cv2.VideoCapture(0)
while True:
 imgPath = urllib.request.urlopen(url)
 imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
 frame = cv2.imdecode(imgNp, -1)
 #ret, frame = video_capture.read()
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = faceCascade.detectMultiScale(gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(60, 60),
  flags=cv2.CASCADE_SCALE_IMAGE)
 
 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 encodings = face_recognition.face_encodings(rgb)
 names = []
 for encoding in encodings:
  matches = face_recognition.compare_faces(data["encodings"],
  encoding)
  name = "Unknown"
  if True in matches:
   matchedIdxs = [i for (i, b) in enumerate(matches) if b]
   counts = {}
   for i in matchedIdxs:
    name = data["names"][i]
    counts[name] = counts.get(name, 0) + 1
   name = max(counts, key=counts.get)


  names.append(name)
  if name != "Unknown":
   if t > 20:
    GPIO.output(PIN, GPIO.LOW)
    t -= 1
   else:
    GPIO.output(PIN, GPIO.HIGH)
    t += 2
  else:
   t = 0
   GPIO.output(PIN, GPIO.LOW)
  for ((x, y, w, h), name) in zip(faces, names):
   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
   cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
   0.75, (0, 255, 0), 2)
 cv2.imshow("Frame", frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
  break
video_capture.release()
cv2.destroyAllWindows()
