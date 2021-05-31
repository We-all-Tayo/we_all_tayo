import math
import numpy as np
import cv2

# real size & distance
REAL_DISTANCE = 2370.0
REAL_HEIGHT = 2000.0
REAL_WIDTH = 1970.0
REAL_DIAGONAL = 2807.294070809113


# 두 포인트 사이의 거리를 반환
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


# 세로 길이의 평균 반환
def calculate_height(points):
    return (euclidean(points[0], points[2]) + euclidean(points[1], points[3])) / 2


# 가로 길이의 평균 반환
def calculate_width(points):
    return (euclidean(points[0], points[1]) + euclidean(points[2], points[3])) / 2


# 대각선 길이의 평균 반환
def calculate_diagonal(points):
    return (euclidean(points[0], points[3]) + euclidean(points[1], points[2])) / 2


# 물체와 카메라가 떨어진 거리 계산
def calculate_distance(alpha, size, real_size):
    return (alpha * real_size) / size


# 물체의 중심점을 계산
def calculate_center(points):
    return (points[0] + points[1] + points[2] + points[3]) / 4


# 두 꼭지점과 각도로 네 좌표 생성
def convert_points(door, radian):
    p1 = np.array((door["x"], door["y"]))
    p4 = np.array((door["x"] + door["w"], door["y"] + door["h"]))
    p2 = np.array((p4[0], p1[1]))
    p3 = np.array((p1[0], p4[1]))
    points = [p1, p2, p3, p4]
    width = calculate_width(points)
    points[0][1] += math.tan(np.pi/2 - radian) * width
    points[2][1] -= math.tan(np.pi/2 - radian) * width
    return points


def calculate_distance_angle(door, radian, image_path):
    # INIT from blue3.jpg
    init_p1 = np.array((1949, 613))
    init_p2 = np.array((3045, 645))
    init_p3 = np.array((1880, 2649))
    init_p4 = np.array((2984, 2695))
    points = [init_p1, init_p2, init_p3, init_p4]
    # print("INIT Points : ", points)

    height = calculate_height(points)
    width = calculate_width(points)
    diagonal = calculate_diagonal(points)

    # get alpha
    alpha_width = (width * REAL_DISTANCE) / REAL_WIDTH
    alpha_hegith = (height * REAL_DISTANCE) / REAL_HEIGHT
    alpha_diagonal = (diagonal * REAL_DISTANCE) / REAL_DIAGONAL
    alpha_mean = (alpha_width + alpha_hegith + alpha_diagonal) / 3

    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    new_points = convert_points(door, radian)
    new_height = calculate_height(new_points)
    distance = calculate_distance(alpha_mean, new_height, REAL_HEIGHT)

    # 이미지에서의 중심점
    size_center = np.array((image_width / 2, image_height / 2))
    # 물체의 중심점 추출
    object_center = calculate_center(new_points)
    # 중심점 거리 계산

    mid_width = euclidean(size_center, object_center)
    print('image center', size_center[0], 'object center', object_center[0])
    if size_center[0] > object_center[0]:
        mid_width *= -1

    angle = math.asin(mid_width / distance)

    # print("distance: ", distance, " angle: ", angle)
    return distance, angle

