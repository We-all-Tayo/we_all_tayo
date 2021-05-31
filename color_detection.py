#NOTE pip install opencv-python imutils numpy
import cv2, imutils
import numpy as np

# color boundaries
LOWER_HSV = {
    'red': np.array([166, 84, 141], np.uint8),
    'green': np.array([40, 122, 145], np.uint8),
    'blue': np.array([100, 122, 117], np.uint8),
    'yellow': np.array([20, 124, 123], np.uint8),
}
UPPER_HSV = {
    'red': np.array([186, 255, 255], np.uint8),
    'green': np.array([70, 225, 225], np.uint8),
    'blue': np.array([120, 255, 255], np.uint8),
    'yellow': np.array([30, 255, 255], np.uint8),
}

class ColorDetection:
    def calculate_area(self, image, image_area, color):
        result = 0
        kernal = np.ones((5, 5), 'uint8')

        # Convert BGR color space to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define masks for each color
        mask = cv2.inRange(hsv_image,
            LOWER_HSV[color], UPPER_HSV[color])

        # Create contour
        mask = cv2.dilate(mask, kernal)
        cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Track color
        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > image_area * 0.001):
                _, _, w, h = cv2.boundingRect(contour)
                result += w*h
        
        return result

    def detect_color(self, image_path, leftup=None, rightdown=None):
        # Get image
        image = cv2.imread(image_path)
        if leftup != None and rightdown != None:
            image = image[leftup[1]: rightdown[1], leftup[0]: rightdown[0]]

        height, width, _ = image.shape
        image_area = height * width

        # Calcuate color areas
        red_area = self.calculate_area(image, image_area, 'red')
        green_area = self.calculate_area(image, image_area, 'green')
        blue_area = self.calculate_area(image, image_area, 'blue')
        yellow_area = self.calculate_area(image, image_area, 'yellow')

        # Red : 6 광역 | Green: 4 지선 | Blue: 3 간선 | Yellow: 5 순환
        areas = {'6': red_area, '4': green_area, '3': blue_area, '5': yellow_area}
        return max(areas, key=lambda x : areas[x])
