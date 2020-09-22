# -*- coding=utf-8 -*-
# python37
import face_recognition

def get_face_feature(face_image):
    image_data = face_recognition.load_image_file(face_image)
    face_locations = face_recognition.face_locations(image_data)
    if len(face_locations) < 0:
        return 0
    else:
        target_face = [face_locations[0], ]
    feature_array = face_recognition.face_encodings(image_data, target_face)
    return feature_array[0]

if __name__ == '__main__':
    print(get_face_feature('FacePhotos/obama_1.jpg'))
