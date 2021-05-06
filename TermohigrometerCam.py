import cv2
import numpy as np
from PIL import Image
import pytesseract
from datetime import datetime, date
import requests
import json

sensor_db_url = 'http://192.168.xxx.xxx:xxxx/send_sensor_data'

def post_to_sensor_db(data):
    date = datetime.now().strftime('%d_%m_%Y')
    time = datetime.now().strftime('%H:%M:%S')
    post1 = json.dumps({'date': date, 'time': time,
                       'type': 'temperature', 'id': '003', 'data': data[0]})
    post2 = json.dumps({'date': date, 'time': time,
                       'type': 'humidity', 'id': '003', 'data': data[2]})
    post3 = json.dumps({'date': date, 'time': time,
                       'type': 'temperature', 'id': '004', 'data': data[1]})
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

def getContours(frame):
    biggest = np.array([])
    maxArea = 0
    frameCanny = cv2.Canny(frame, 45, 55)
    frameCanny = cv2.dilate(frameCanny, kernel, iterations=3)
    frameCanny = cv2.erode(frameCanny, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(
        frameCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 7000 and area < 80000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    diff = np.diff(myPoints, axis=1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def getWarp(img, ptsSqu):
    ptsSqu = reorder(ptsSqu)
    imgOut = np.array([])
    pts1 = np.float32(ptsSqu)
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOut = cv2.warpPerspective(img, matrix, (300, 300))
    return imgOut

#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel = np.ones((3, 3))
cap = cv2.VideoCapture('/dev/webcam_port')
digitsPos = [[[7, 64], [0, 92]], [[63, 118], [0, 92]], [[124, 175], [0, 92]], [[7, 64], [92, 184]], [
    [63, 118], [92, 184]], [[124, 178], [92, 184]], [[4, 62], [184, 275]], [[69, 134], [184, 275]]]
segmentPos = [[[48, 65], [12, 49]], [[43, 62], [49, 84]], [[14, 53], [73, 92]], [
    [3, 23], [49, 84]], [[3, 23], [12, 49]], [[19, 53], [2, 25]], [[19, 53], [40, 60]]]
DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): "0",
    (1, 1, 0, 0, 0, 0, 0): "1",
    (1, 0, 1, 1, 0, 1, 1): "2",
    (1, 1, 1, 0, 0, 1, 1): "3",
    (1, 1, 0, 0, 1, 0, 1): "4",
    (0, 1, 1, 0, 1, 1, 1): "5",
    (0, 1, 1, 1, 1, 1, 1): "6",
    (1, 1, 0, 0, 0, 1, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 0, 1, 1, 1): "9", }
displayDetected = False
while not displayDetected:
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (3, 3), 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    #frame = clahe.apply(frame)
    ptsSqr = getContours(frame)
    if ptsSqr.size != 0:
        displayDetected = True
        frame = getWarp(frame, ptsSqr)[5:-15, 80:-25]
frame = preprocess(frame, 43, 7)
digits = []
for pos in digitsPos:
    digits.append(frame[pos[1][0]:pos[1][1], pos[0][0]:pos[0][1]])
dataRaw = []
for i in range(0, len(digits)):
    digits[i] = cv2.resize(digits[i], (65, 92))
    #dataRaw.append(pytesseract.image_to_string(digits[i], config="--psm 10 digits"))
    #dataRaw.append(pytesseract.image_to_string(digits[i], config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')[0])
    segments = []
    for seg in segmentPos:
        segAvg = np.mean(digits[i][seg[1][0]:seg[1][1], seg[0][0]:seg[0][1]])
        if (segAvg < 180):
            segments.append(1)
            cv2.rectangle(digits[i], (seg[0][0], seg[1][0]),
                          (seg[0][1], seg[1][1]), (0, 0, 0), 1)
        else:
            segments.append(0)
    segments = tuple(segments)
    try:
        dataRaw.append(DIGITS_LOOKUP[segments])
    except:
        dataRaw.append("E")
data = [0, 0, 0]
if not "E" in dataRaw:
    dataRaw = "".join(dataRaw)
    data[0] = dataRaw[0:2]+"."+dataRaw[2]
    data[1] = dataRaw[3:5]+"."+dataRaw[5]
    data[2] = dataRaw[6:]
print(data)
post_to_sensor_db(data)
imgStack = np.vstack(tuple(digits))
#cv2.imshow("outImage", imgStack)
#cv2.imshow('frame', frame)
#if (cv2.waitKey(10000) & 0xFF == ord('q')):
#        break
cv2.waitKey(1000)
cap.release()
cv2.destroyAllWindows()