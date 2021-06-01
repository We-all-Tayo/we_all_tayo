import cv2
import numpy as np
import tensorflow as tf

# import core.utils as utils

IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

class Yolo:
    def __init__(self, infer):
        self.infer = infer

    def yolo(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape

        img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_input = img_input / 255.
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        img_input = tf.constant(img_input)

        pred_bbox = self.infer(img_input)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # boxes: 버스 테두리
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=SCORE_THRESHOLD
        )

        bus_place, bus_area = [], []
        door_place, door_area = [], []
        route_number_place, route_number_area = [], []
        bus_number_place, bus_number_area = [], []

        for i in range(len(classes)):
            for j in range(len(classes[i])):
                # classess 0 : bus, 1: bus door, 2: route number, 3: bus number
                if classes[i][j] == 0 and scores[i][j] >= 0.9:
                    bus = boxes[i][j].numpy()
                    bus_place.append(bus)
                    bus_area.append((bus[3] - bus[1]) * (bus[2] - bus[0]))
                elif classes[i][j] == 1 and scores[i][j] >= 0.9:
                    door = boxes[i][j].numpy()
                    door_place.append(door)
                    door_area.append((door[3] - door[1]) * (door[2] - door[0]))
                elif classes[i][j] == 2 and scores[i][j] >= 0.9:
                    route_number = boxes[i][j].numpy()
                    route_number_place.append(route_number)
                    route_number_area.append((route_number[3] - route_number[1]) * (route_number[2] - route_number[0]))
                elif classes[i][j] == 3 and scores[i][j] >= 0.9:
                    bus_number = boxes[i][j].numpy()
                    bus_number_place.append(bus_number)
                    bus_number_area.append((bus_number[3] - bus_number[1]) * (bus_number[2] - bus_number[0]))

        bus_index = bus_area.index(max(bus_area))
        door_index = door_area.index(max(door_area))
        route_number_index = route_number_area.index(max(route_number_area))
        bus_number_index = bus_number_area.index(max(bus_number_area))

        # (y0, x0, y1, x1)
        bus_place[bus_index][0] *= height
        bus_place[bus_index][2] *= height
        bus_place[bus_index][1] *= width
        bus_place[bus_index][3] *= width

        door_place[door_index][0] *= height
        door_place[door_index][2] *= height
        door_place[door_index][1] *= width
        door_place[door_index][3] *= width

        route_number_place[route_number_index][0] *= height
        route_number_place[route_number_index][2] *= height
        route_number_place[route_number_index][1] *= width
        route_number_place[route_number_index][3] *= width

        bus_number_place[bus_number_index][0] *= height
        bus_number_place[bus_number_index][2] *= height
        bus_number_place[bus_number_index][1] *= width
        bus_number_place[bus_number_index][3] *= width

        bus_leftup = (int(bus_place[bus_index][1]), int(bus_place[bus_index][0]))
        bus_rightdown = (int(bus_place[bus_index][3]), int(bus_place[bus_index][2]))
        route_number_leftup = (int(route_number_place[route_number_index][1]), int(route_number_place[route_number_index][0]))
        route_number_rightdown = (int(route_number_place[route_number_index][3]), int(route_number_place[route_number_index][2]))
        bus_number_leftup = (int(bus_number_place[bus_number_index][1]), int(bus_number_place[bus_number_index][0]))
        bus_number_rightdown = (int(bus_number_place[bus_number_index][3]), int(bus_number_place[bus_number_index][2]))

        door = {}
        door["x"] = int(door_place[door_index][1])
        door["y"] = int(door_place[door_index][0])
        door["w"] = int(door_place[door_index][3]) - int(door_place[door_index][1])
        door["h"] = int(door_place[door_index][2]) - int(door_place[door_index][0])

        return bus_leftup, bus_rightdown, door, route_number_leftup, route_number_rightdown, bus_number_leftup, bus_number_rightdown
