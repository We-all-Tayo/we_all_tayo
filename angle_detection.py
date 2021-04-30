import math
import cv2
import numpy as np
import statistics

def detect_angle(image_path, door):
    image = cv2.imread(image_path)
    height, _, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90,
                            minLineLength=10, maxLineGap=100)

    angles = []
    for line in lines:
        # 왼쪽 위 꼭지점 (line[0][0], line[0][1]), 오른쪽 아래 꼭지점 (line[0][2], line[0][3])
        if 0 < line[0][1] < height*0.117 and 0 < line[0][3] < height*0.117:
            if line[0][3] == line[0][1]:
                continue
            angle = math.atan((line[0][0] - line[0][2]) / (line[0][3] - line[0][1]))
            if 0 < angle < (np.pi * 180 / 4):
                angles.append(round(angle, 2))

    return statistics.mode(angles)
