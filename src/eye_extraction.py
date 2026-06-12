import cv2 
import numpy as np




def Right_resised_image(frame,result):
    if result==None:
        return None
    if result.face_landmarks is None or len(result.face_landmarks) == 0:
        return None
    Right_Eye_indices=[263,
    362,
    387,
    386,
    385,
    384,
    398,
    373,
    374,
    380,
    381,
    382]
    ih,iw,i_d=frame.shape
    X_coordinates_left=[int(result.face_landmarks[0][i].x *iw ) for i in Right_Eye_indices]
    Y_coordinate_left=[int(result.face_landmarks[0][i].y *ih ) for i in Right_Eye_indices]
    x_minl, x_maxl = min(X_coordinates_left), max(X_coordinates_left)
    y_minl, y_maxl = min(Y_coordinate_left), max(Y_coordinate_left)
    width_of_left_eye=x_maxl-x_minl
    hight_of_left_eye=y_maxl-y_minl
    x_minl -= 0.15 * width_of_left_eye
    x_maxl += 0.15 * width_of_left_eye
    y_minl -= 0.25 * hight_of_left_eye
    y_maxl += 0.25 * hight_of_left_eye
    x_minl = int(x_minl)
    x_maxl = int(x_maxl)
    y_minl = int(y_minl)
    y_maxl = int(y_maxl)
    x_minl = max(0, x_minl)
    y_minl = max(0, y_minl)
    x_maxl = min(iw, x_maxl)
    y_maxl = min(ih, y_maxl)
    left_eye_crop = frame[y_minl:y_maxl, x_minl:x_maxl]
    resized_left_eye=cv2.resize(left_eye_crop,(32,32))
    return resized_left_eye


def left_resised_image(frame,result):
    if result==None:
        return None
    if result.face_landmarks is None or len(result.face_landmarks) == 0:
        return None
    Left_Eye_indices=[33,
    133,
    160,
    159,
    158,
    157,
    173,
    144,
    145,
    153,
    154,
    155]
    ih,iw,i_d=frame.shape
    X_coordinates_left=[int(result.face_landmarks[0][i].x *iw ) for i in Left_Eye_indices]
    Y_coordinate_left=[int(result.face_landmarks[0][i].y *ih ) for i in Left_Eye_indices]
    x_minl, x_maxl = min(X_coordinates_left), max(X_coordinates_left)
    y_minl, y_maxl = min(Y_coordinate_left), max(Y_coordinate_left)
    width_of_left_eye=x_maxl-x_minl
    hight_of_left_eye=y_maxl-y_minl
    x_minl -= 0.15 * width_of_left_eye
    x_maxl += 0.15 * width_of_left_eye
    y_minl -= 0.25 * hight_of_left_eye
    y_maxl += 0.25 * hight_of_left_eye
    x_minl = int(x_minl)
    x_maxl = int(x_maxl)
    y_minl = int(y_minl)
    y_maxl = int(y_maxl)
    x_minl = max(0, x_minl)
    y_minl = max(0, y_minl)
    x_maxl = min(iw, x_maxl)
    y_maxl = min(ih, y_maxl)
    left_eye_crop = frame[y_minl:y_maxl, x_minl:x_maxl]
    resized_left_eye=cv2.resize(left_eye_crop,(32,32))
    return resized_left_eye



def combined_image(frame,result):
    if result==None:
        return None
    if result.face_landmarks is None or len(result.face_landmarks) == 0:
        return None
    left_eye=left_resised_image(frame,result)
    right_eye=Right_resised_image(frame,result)
    combined = np.hstack((left_eye, right_eye))
    return combined
    