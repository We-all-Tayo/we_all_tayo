import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


def detect_angle(img, door):
    src = cv2.imread(img)
    dst = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=10, maxLineGap=100)

    angles = []
    # (302, 103) (349, 298)
    for i in lines:
        # 왼쪽 위 꼭지점 (i[0][0], i[0][1]), 오른쪽 아래 꼭지점 (i[0][2], i[0][3])
        if 0 < i[0][1] < 300 and 0 < i[0][3] < 300:
            if i[0][3] == i[0][1]:
                continue
            angle = math.atan((i[0][0] - i[0][2]) / (i[0][3] - i[0][1]))
            if 0 < angle < (np.pi * 180 / 4):
                angles.append(round(angle, 2))
                cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)

    # print(angles)

    prob_angle = mode(angles)
    # print(prob_angle)
    plt.imshow(dst)
    return angle
