from bus_arrive import BusArrive
from color_detection import ColorDetection
from number_detection import NumberDetection
from routenumber_detection import RouteNumberDetection
from door_detection import DoorDetection
from angle_detection import AngleDetection
from calculate_distance_angle import Calculator
from yolo import Yolo

#import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


# gpu 사용시 해제해주세요.
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)



MODEL_PATH = "./checkpoints/yolov4-last"


def main(target_bus, target_station, img_path):
    # load model
    saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]

    bus_arrive = BusArrive()
    color_detection = ColorDetection()
    number_detection = NumberDetection()
    route_number_detection = RouteNumberDetection()
    door_detection = DoorDetection()
    angle_detection = AngleDetection()
    calculator = Calculator()
    yolo = Yolo(infer)

    # call 공공API
    #bus_dict = bus_arrive.get_bus_dict(target_station)
    bus_dict = {
        # 3번이 파란색
        "370": ("3", None),
        "7212": ("4", None),
        "642": ("3", None),
        "4211": ("4", "서울 74사 4226"),
    }

    if target_bus not in bus_dict:
        return "Target bus is not comming."

    target_color, plain_no = bus_dict[target_bus]

    same_color = 0
    diff_color = 0
    for route, (color, _) in bus_dict.items():
        if color == target_color:
            same_color += 1
        else:
            diff_color += 1

    # YOLO
    bus_leftup, bus_rightdown, door_dict, route_number_leftup, route_number_rightdown, bus_number_leftup, bus_number_rightdown = yolo.yolo(img_path)

    if diff_color > 0:
        detected_color = color_detection.detect_color(img_path, leftup=bus_leftup, rightdown=bus_rightdown)
        if detected_color != target_color:
            return "Unexpected Color"

    plain_no = plain_no[-4:]
    print("plain_no", plain_no)
    if same_color > 1 and (number_detection.detect_number(img_path, plain_no, bus_number_leftup, bus_number_rightdown) == False \
                and route_number_detection.detect_routenumber(img_path, target_bus,route_number_leftup, route_number_rightdown) == False):
        return "Unexpected Number"

    # Door Detection OpenCV
    door = door_detection.detect_door(img_path, bus_leftup, bus_rightdown)

    # Angle Detection
    radian = angle_detection.detect_angle(img_path, bus_leftup, bus_rightdown)

    door_union = {}
    door_union["x"] = min(door['x'], door_dict['x'])
    door_union["y"] = min(door['y'], door_dict['y'])
    door_union["right_x"] = max(door['x'] + door['w'], door_dict['x'] + door_dict['w'])
    door_union["down_y"] = max(door['y'] + door['h'], door_dict['y'] + door_dict['h'])
    door_union["w"] = door_union["right_x"] - door_union["x"]
    door_union["h"] = door_union["down_y"] - door_union["y"]

    door_intersect = {}
    door_intersect["x"] = max(door['x'], door_dict['x'])
    door_intersect["y"] = max(door['y'], door_dict['y'])
    door_intersect["right_x"] = min(door['x'] + door['w'], door_dict['x'] + door_dict['w'])
    door_intersect["down_y"] = min(door['y'] + door['h'], door_dict['y'] + door_dict['h'])
    door_intersect["w"] = door_intersect["right_x"] - door_intersect["x"]
    door_intersect["h"] = door_intersect["down_y"] - door_intersect["y"]

    # Calculation
    distance1, angle1 = calculator.calculate_distance_angle(door, radian, img_path)  # opencv
    distance2, angle2 = calculator.calculate_distance_angle(door_dict, radian, img_path)  # yolo
    distance3, angle3 = calculator.calculate_distance_angle(door_union, radian, img_path)  # opencv + yolo union
    distance4, angle4 = calculator.calculate_distance_angle(door_intersect, radian, img_path)  # opencv + yolo intersect

    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.rectangle(img, (door['x'], door['y']), (door['x'] + door['w'], door['y'] + door['h']), color=(255, 0, 0),
    #               thickness=5)  # blue
    # cv2.rectangle(img, (door_dict['x'], door_dict['y']),
    #               (door_dict['x'] + door_dict['w'], door_dict['y'] + door_dict['h']), color=(0, 255, 0),
    #               thickness=5)  # green
    # cv2.rectangle(img, (door_union['x'], door_union['y']),
    #               (door_union['x'] + door_union['w'], door_union['y'] + door_union['h']), color=(0, 0, 255),
    #               thickness=5)  # red
    # cv2.rectangle(img, (door_intersect['x'], door_intersect['y']),
    #               (door_intersect['x'] + door_intersect['w'], door_intersect['y'] + door_intersect['h']),
    #               color=(255, 0, 255), thickness=5)  # pink
    # cv2.imwrite("test.png", img)

    print(str(round(distance1 / 1000, 2)) + " meter")
    print(str(round(angle1 * 180 / np.pi)) + " 도")
    print(str(round(distance2 / 1000, 2)) + " meter")
    print(str(round(angle2 * 180 / np.pi)) + " 도")
    print(str(round(distance3 / 1000, 2)) + " meter")
    print(str(round(angle3 * 180 / np.pi)) + " 도")
    print(str(round(distance4 / 1000, 2)) + " meter")
    print(str(round(angle4 * 180 / np.pi)) + " 도")

    return (
        str(round(distance1 / 1000, 2)) + " meter",
        str(round(angle1 * 180 / np.pi)) + " 도",
    )

if __name__ == '__main__':
    print(
        "RESULT:",
        main(target_bus="4211", target_station="23322", img_path="./input/bus4211.jpg"),
    )
