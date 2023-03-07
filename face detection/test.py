import face_recognition
import os
import cv2
import numpy as np
import json

known_face_names = []
known_face_encodings = []

def getPictures(directory):
	files = os.listdir(directory)
	for img in files:
		(name, _) = img.split('.')
		face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(directory + img))[0]
		known_face_encodings.append(face_encoding)
		known_face_names.append(name)
		print("__getPictures__")
		
def getUnknown(directory):
	files = os.listdir(directory)
	data = []
	for img in files:
		name = "unknown"
		rgb_frame = cv2.cvtColor(cv2.imread(directory + img), cv2.COLOR_BGR2RGB)
		face_locations = face_recognition.face_locations(rgb_frame)
		face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
		for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:name = known_face_names[best_match_index]
			print(str(name) + " ~ " + str(img))

			if name != "unknown":
				flag = True
				for element in data:
					if element == name:
						flag = False
						break
						
				if flag: data.append(name)
			
				print(data)


getPictures("face/")
getUnknown("unknown/")
