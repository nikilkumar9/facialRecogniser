import cv2
import face_recognition as fr
import os
import sys
import numpy
from matplotlib import pyplot as plt

path = r'/Users/NickKumar/EE_training/face_db/'
# image = plt.imread(os.path.join(path, 'family.jpg'))

family_image = fr.api.load_image_file('/Users/NickKumar/EE_training/face_db/family.jpg')
nithin_image = fr.api.load_image_file('/Users/NickKumar/EE_training/face_db/rajni.jpg')

face_locations = fr.api.face_locations(family_image)

encodings = fr.api.face_encodings(family_image)
nithin_encodings = fr.api.face_encodings(nithin_image)

dis = fr.api.face_distance(encodings, nithin_encodings[0])
matches = fr.api.compare_faces(encodings, nithin_encodings[0], tolerance = 0.4)

print(dis)
print(match)

font = cv2.FONT_HERSHEY_SIMPLEX
for top, right, bottom, left in face_locations:
    cv2.rectangle(family_image, (left, top), (right, bottom), (255, 0, 200), 2, cv2.LINE_AA)
    if True in matches:
        cv2.putText(family_image, 'count', (bottom, left), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

plt.xticks([], [])
plt.yticks([], [])
plt.imshow(family_image)
plt.show()