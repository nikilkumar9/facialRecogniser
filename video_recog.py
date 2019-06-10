import cv2
import face_recognition as fr
import os
import sys
import numpy
from matplotlib import pyplot as plt
import time

db_path = '/Users/NickKumar/facialRecognition/face_db'
file_names = os.listdir(db_path)

# Creating a list of the names/titles of known faces (array known_face_names)
known_faces_names = list()

# Creating a list to store the face encodings of the known faces after computing the face encodings
known_face_encodings = list()

for file_name in file_names:
    # Adding unedited names to list of face names.
    if (file_name == '.DS_Store' or file_name == '.DS_STORE'):
        continue
    known_faces_names.append(file_name.split('.')[0])

    # Adding face encodings to detected faces
    name = os.path.join(db_path, file_name)
    image = fr.api.load_image_file(name, mode='RGB')
    face_encoding = fr.api.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)

print(known_faces_names)

# Turn on Video Camera / Web Cam
vs = cv2.VideoCapture(0)
time.sleep(1)

program_run = True
while True:
    if (program_run): # Search for faces in every other frame for increased speed
        face, frame = vs.read()
        small_frame = cv2.resize(frame, (0,0), fx=1/5, fy=1/5)
        
        # Create a list of face encodings from each frame
        face_locations = fr.api.face_locations(small_frame)
        frame_face_encodings = fr.api.face_encodings(small_frame)

        for i, frame_face_encoding in enumerate(frame_face_encodings):
            matches = fr.api.compare_faces(known_face_encodings, frame_face_encoding, tolerance = 0.6) # Return a list of matches faces if True.

            if True in matches:
                index = matches.index(True) # Finding the first true comparison of a known face
                name = known_faces_names[index]
            else:
                name = "Unknown Person"

        # For each face detected in given frame, compare the faces with the encodings of known faces to check if same person.
            top, right, bottom, left = face_locations[i]
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            cv2.rectangle(frame, (left,top), (right,bottom), (0, 255, 0), 2,cv2.LINE_AA)
            cv2.putText(frame, name, (left , bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
        
        cv2.imshow('Frame', frame)

    program_run = not program_run

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


vs.release()
cv2.destroyAllWindows()
