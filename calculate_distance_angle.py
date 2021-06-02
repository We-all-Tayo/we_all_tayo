import math
import numpy as np
import cv2

# real size & distance
REAL_DISTANCE = 2370.0
REAL_HEIGHT = 2000.0
REAL_WIDTH = 1060.0
REAL_DIAGONAL = 2263.53705514
CORRECTION = 1.2

class Calculator:

    def __init__(self):
        init_p1 = np.array((1949, 613))
        init_p2 = np.array((3045, 645))
        init_p3 = np.array((1880, 2649))
        init_p4 = np.array((2984, 2695))
        points = [init_p1, init_p2, init_p3, init_p4]

        height = (np.linalg.norm(points[0] - points[2]) + np.linalg.norm(points[1] - points[3])) / 2
        width = (np.linalg.norm(points[0] - points[1]) + np.linalg.norm(points[2] - points[3])) / 2
        diagonal = (np.linalg.norm(points[0] - points[3]) + np.linalg.norm(points[1] - points[2])) / 2

        # get alpha
        alpha_width = width * REAL_DISTANCE / REAL_WIDTH
        alpha_height = height * REAL_DISTANCE / REAL_HEIGHT
        alpha_diagonal = diagonal * REAL_DISTANCE / REAL_DIAGONAL
        self.alpha_mean = (alpha_width + alpha_height + alpha_diagonal) / 3 * CORRECTION


    # 두 꼭지점과 각도로 네 좌표 생성
    def convert_points(self, door, radian):
        p1 = np.array((door["x"], door["y"]))
        p4 = np.array((door["x"] + door["w"], door["y"] + door["h"]))
        p2 = np.array((p4[0], p1[1]))
        p3 = np.array((p1[0], p4[1]))

        points = [p1, p2, p3, p4]

        points[0][1] += math.tan(np.pi/2 - radian) * door["w"]
        points[2][1] -= math.tan(np.pi/2 - radian) * door["w"]
        return points


    def calculate_distance_angle(self, door, radian, image_path):
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        new_points = self.convert_points(door, radian)
        new_height = (np.linalg.norm(new_points[0] - new_points[2]) + np.linalg.norm(new_points[1] - new_points[3])) / 2
        distance = (self.alpha_mean * REAL_HEIGHT) / new_height

        # 이미지에서의 중심점
        size_center = np.array((image_width / 2, image_height / 2))
        # 물체의 중심점 추출
        object_center = (new_points[0] + new_points[1] + new_points[2] + new_points[3]) / 4
        # 중심점 거리 계산
        mid_width = np.linalg.norm(size_center - object_center)
        print('image center', size_center[0], 'object center', object_center[0])
        if size_center[0] > object_center[0]:
            mid_width *= -1

        angle = math.asin(mid_width / distance)

        return distance, angle
