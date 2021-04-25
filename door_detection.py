import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

plt.style.use('dark_background')


def detect_door(img, leftup, rightdown):
    img_ori = cv2.imread(img)
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    height, width, channel = img_ori.shape
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours,
                     contourIdx=-1, color=(255, 255, 255))
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 255, 255), thickness=2)

        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 4, 16
    MIN_RATIO, MAX_RATIO = 0.01, 0.5
    RIGHT_X = (rightdown[0] - leftup[0]) / 2
    MAX_X = rightdown[0] * 0.8  # 문이 바운더리 밖에 있지는 않으니까

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO \
                and MAX_X > d["x"] > RIGHT_X:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # print(possible_contours[0]["x"])

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)
    # print(possible_contours[0])

    possible_contours.sort(key=lambda x: x["h"], reverse=True)
    prob_door = possible_contours[0]
    print((prob_door['x'], prob_door["y"]), (prob_door['x'] +
          prob_door['w'], prob_door['y'] + prob_door['h']))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.rectangle(temp_result, pt1=(prob_door['x'], prob_door['y']),
                  pt2=(prob_door['x'] + prob_door['w'], prob_door['y'] + prob_door['h']), color=(255, 255, 255),
                  thickness=2)

    cv2.rectangle(img_ori, pt1=(prob_door['x'], prob_door['y']),
                  pt2=(prob_door['x'] + prob_door['w'], prob_door['y'] + prob_door['h']), color=(255, 255, 255),
                  thickness=2)

    return prob_door
