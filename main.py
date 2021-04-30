from bus_arrive import get_bus_dict
from color_detection import detect_color
from number_detection import detect_number
from door_detection import detect_door
from angle_detection import detect_angle
from calculate_distance_angle import calculate_distance_angle
from yolo import yolo

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

MODEL_PATH = "./checkpoints/yolov4-416"


def main(target_bus, target_station, img_path):
    # load model
    saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]

    # call 공공API
    # XXX bus_dict = get_bus_dict(target_station)
    bus_dict = {
        # 3번이 파란색
        "370": "3",
        "7212": "4",
        "642": "3",
        "4211": "4",
    }

    if target_bus not in bus_dict:
        return "Target bus is not comming."

    target_color = bus_dict[target_bus]
    same_color = 0
    diff_color = 0
    for route, color in bus_dict.items():
        if color == target_color:
            same_color += 1
        else:
            diff_color += 1

    # YOLO
    leftup, rightdown = yolo(infer, img_path)

    if diff_color > 0:
        detected_color = detect_color(img_path, leftup=leftup, rightdown=rightdown)
        if detected_color != target_color:
            return "Unexpected Color"

    if (
        same_color > 1
        and detect_number(img_path, target_bus, leftup, rightdown) == False
    ):
        return "Unexpected Number"

    # Door Detection
    door = detect_door(img_path, leftup, rightdown)

    # Angle Detection
    radian = detect_angle(img_path, door)

    # Calculation
    distance, angle = calculate_distance_angle(door, radian)

    return (
        str(round(distance / 1000, 2)) + " meter",
        str(round(angle * 180 / np.pi)) + " 도",
    )

print(
    "RESULT:",
    main(target_bus="4211", target_station="23322", img_path="./input/bus4211.jpg"),
)

