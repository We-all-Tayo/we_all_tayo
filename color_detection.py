#NOTE pip install opencv-python imutils numpy
import cv2, imutils
import numpy as np

# color boundaries
LOWER_HSV = {
    'red': np.array([166, 84, 141], np.uint8),
    'green': np.array([40, 122, 145], np.uint8),
    'blue': np.array([97, 100, 117], np.uint8),
    'yellow': np.array([20, 124, 123], np.uint8),
}
UPPER_HSV = {
    'red': np.array([186, 255, 255], np.uint8),
    'green': np.array([60, 225, 225], np.uint8),
    'blue': np.array([140, 255, 255], np.uint8),
    'yellow': np.array([30, 255, 255], np.uint8),
}

def detect_color(image_path):
    # Get image and convert BGR color space to HSV color space
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color areas
    red_area = 0
    green_area = 0
    blue_area = 0
    yellow_area = 0

    # Define masks for each color
    red_mask = cv2.inRange(hsv_image,
        LOWER_HSV['red'], UPPER_HSV['red'])
    green_mask = cv2.inRange(hsv_image,
        LOWER_HSV['green'], UPPER_HSV['green'])
    blue_mask = cv2.inRange(hsv_image,
        LOWER_HSV['blue'], UPPER_HSV['blue'])
    yellow_mask = cv2.inRange(hsv_image,
        LOWER_HSV['yellow'], UPPER_HSV['yellow'])

    kernal = np.ones((5, 5), 'uint8')

    # Red: Create contour
    red_mask = cv2.dilate(red_mask, kernal)
    cv2.bitwise_and(image, image, mask=red_mask)
    contours, hierarchy = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Red: Track color
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            _, _, w, h = cv2.boundingRect(contour)
            red_area += w*h

    # Green: Create contour
    green_mask = cv2.dilate(green_mask, kernal)
    cv2.bitwise_and(image, image, mask=green_mask)
    contours, hierarchy = cv2.findContours(
        green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Green: Track color
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            _, _, w, h = cv2.boundingRect(contour)
            green_area += w*h

    # Blue: Create contour
    blue_mask = cv2.dilate(blue_mask, kernal)
    cv2.bitwise_and(image, image, mask=blue_mask)
    contours, hierarchy = cv2.findContours(
        blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Blue: Track color
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            _, _, w, h = cv2.boundingRect(contour)
            blue_area += w*h

    # Yellow: Create contour
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    cv2.bitwise_and(image, image, mask=yellow_mask)
    contours, hierarchy = cv2.findContours(
        yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Yellow: Track color
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            _, _, w, h = cv2.boundingRect(contour)
            yellow_area += w*h

    # Red : 6 광역 | Green: 4 지선 | Blue: 3 간선 | Yellow: 5 순환
    areas = {'6': red_area, '4': green_area, '3': blue_area, '5': yellow_area}
    return max(areas, key=lambda x : areas[x])