import math
import numpy as np

# real size & distance
REAL_DISTANCE = 247.6
REAL_HEIGHT = 85.60
REAL_WIDTH = 53.98
REAL_DIAGONAL = math.sqrt(math.pow(REAL_HEIGHT, 2) + math.pow(REAL_WIDTH, 2))


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
def convert_points(p1, p4, radian):
    p2 = np.array((p4[0], p1[1]))
    p3 = np.array((p1[0], p4[1]))
    points = [p1, p2, p3, p4]
    width = calculate_width(points)
    points[0][1] += math.tan(radian) * width
    points[2][1] -= math.tan(radian) * width
    return points


def calculate_distance_angle(p1, p4, radian):
    # INIT
    p1 = np.array((303, 359))
    p2 = np.array((476, 360))
    p3 = np.array((301, 642))
    p4 = np.array((477, 643))
    points = [p1, p2, p3, p4]
    print("p: ", points)

    height = calculate_height(points)
    width = calculate_width(points)
    diagonal = calculate_diagonal(points)

    # get alpha
    alpha_width = (width * REAL_DISTANCE) / REAL_WIDTH
    alpha_hegith = (height * REAL_DISTANCE) / REAL_HEIGHT
    alpha_diagonal = (diagonal * REAL_DISTANCE) / REAL_DIAGONAL
    alpha_mean = (alpha_width + alpha_hegith + alpha_diagonal) / 3

    new_points = convert_points(p1, p4, radian)
    new_height = calculate_height(new_points)
    # new_width = calculate_width(new_points)
    # new_diagonal = calculate_diagonal(new_points)
    distance = calculate_distance(alpha_mean, new_height, REAL_HEIGHT)

    # 이미지에서의 중심점
    # TODO: IMAGE_WIDTH, IMAGE_HEIGHT
    size_center = np.array((384, 512))
    # 물체의 중심점 추출
    object_center = calculate_center(new_points)
    # 중심점 거리 계산
    mid_width = euclidean(size_center, object_center)
    angle = math.atan(mid_width / distance)

    print("distance: ", distance, " angle: ", angle)
    return distance, angle