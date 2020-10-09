import cv2
import numpy as np

def img_preprocessor(img):
    return normalize(face_cropper(img))


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def face_cropper(img):
    '''
    Finds face and crops image
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    factor = 1.05 # scale factor
    Found = False # is face found

    for i in range(5):
        face = face_detector.detectMultiScale(img_gray, factor - i/100., 5)
        if not len(face)==0:
            # if face was found
            face_x, face_y, face_w, face_h = face[0]

            # make it a bit smaller:
            new_h = face_h * 0.8
            new_w = face_w * 0.7
            face_y += (face_h-new_h) / 2
            face_x += (face_w - new_w) / 2
            face_h, face_w = new_h, new_w

            img_cropped = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
            Found = True
            break

    if not Found:
        print('Face was not recognized')
        img_cropped = img

    return img_cropped

def normalize(img):
    '''
    resize image to (175, 200) with different algorithm if it's downscaled or upscaled
    '''
    if img.shape[0] > 200:
        img_resized = cv2.resize(img, (175, 200), interpolation = cv2.INTER_AREA) # downscale
    elif img.shape[0] < 200:
        img_resized = cv2.resize(img, (175, 200), interpolation = cv2.INTER_CUBIC) # upscale
    return img_resized
