import cv2
import numpy as np
from PIL import Image
import pytesseract
from datetime import datetime, date
import requests
import json

sensor_db_url = #############
DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}
H_W_Ratio = 1.9
arc_tan_theta = 6.0


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    for i in range(len(one_d_array)):
        if one_d_array[i] < 12 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


def find_digits_positions(img, reserved_threshold=20):
    digits_positions = []
    img_array = np.sum(img, axis=0)
    horizon_position = helper_extract(img_array, threshold=reserved_threshold)
    img_array = np.sum(img, axis=1)
    vertical_position = helper_extract(
        img_array, threshold=reserved_threshold * 4)
    # make vertical_position has only one element
    if len(vertical_position) > 1:
        vertical_position = [
            (vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
    for h in horizon_position:
        for v in vertical_position:
            digits_positions.append(list(zip(h, v)))
    assert len(digits_positions) > 0, "Failed to find digits's positions"

    return digits_positions


def post_to_sensor_db(data):
    date = datetime.now().strftime('%d_%m_%Y')
    time = datetime.now().strftime('%H:%M:%S')
    post1 = json.dumps({'date': date, 'time': time,
                       'type': 'temperature', 'id': '003', 'data': data[0]})
    post2 = json.dumps({'date': date, 'time': time,
                       'type': 'humidity', 'id': '003', 'data': data[0]})
    post3 = json.dumps({'date': date, 'time': time,
                       'type': 'temperature', 'id': '004', 'data': data[0]})
    posts = [post1, post2, post3]
    for post in posts:
        try:
            response = requests.post(sensor_db_url, post)
            print('sensor db response: ' + str(response))
        except:
            print('conection no established')


def preprocess(img, block_size, gamma):
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, gamma)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


width, height = 170, 50
ptsTin = np.float32([[271, 129], [393, 146], [259, 192], [380, 206]])
ptsTout = np.float32([[259, 189], [383, 204], [245, 251], [370, 271]])
ptsHin = np.float32([[250, 255], [351, 264], [249, 322], [342, 326]])
ptsF = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrixTin = cv2.getPerspectiveTransform(ptsTin, ptsF)
matrixTout = cv2.getPerspectiveTransform(ptsTout, ptsF)
matrixHin = cv2.getPerspectiveTransform(ptsHin, ptsF)
kernelv = np.array(((0, 1, 0), (0, 1, 0), (0, 1, 0)), np.uint8)
kernelh = np.array(((0, 0, 0), (1, 1, 1), (0, 0, 0)), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

cap = cv2.VideoCapture('/dev/webcam_port')
ret, frame = cap.read()
frame = cv2.GaussianBlur(frame, (3, 3), 3)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
#frame = clahe.apply(frame)
#cv2.imshow('frame', frame)

imgTIn = cv2.warpPerspective(frame, matrixTin, (width, height))
imgTIn = preprocess(imgTIn, 23, 7)
digits_positions = find_digits_positions(imgTIn)
print(digits_positions)

imgTOut = cv2.warpPerspective(frame, matrixTout, (width, height))
imgTOut = preprocess(imgTOut, 23, 5)

imgHIn = cv2.warpPerspective(frame, matrixHin, (width, height))
imgHIn = cv2.resize(imgHIn, (width, height*2))
imgHIn = preprocess(imgHIn, 33, 5)

tIn = pytesseract.image_to_string(imgTIn, config="--psm 7 digits")
tOut = pytesseract.image_to_string(imgTOut, config="--psm 7 digits")
hIn = pytesseract.image_to_string(imgHIn, config="--psm 7 digits")

print(f'Temperatura interna: {tIn}')
print(f'Temperatura externa: {tOut}')
print(f'Humedad interna: {hIn} % \n')

data = [10, 10, 10]
post_to_sensor_db(data)

imgStack = np.vstack((imgTIn, imgTOut, imgHIn))
cv2.imshow("outImage", imgStack)
while True:
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
