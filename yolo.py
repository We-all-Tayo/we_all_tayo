import cv2
import numpy as np
import tensorflow as tf
# import core.utils as utils

IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

def yolo(infer, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape

    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_input = img_input / 255.
    img_input = img_input[np.newaxis, ...].astype(np.float32)
    img_input = tf.constant(img_input)

    pred_bbox = infer(img_input)

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

    bus_place = np.zeros(4)

    for i in range(len(classes)):
        for j in range(len(classes[i])):
            # class 5 : bus
            if classes[i][j] == 0:
                bus_place += boxes[i][j].numpy()

    # (y0, x0, y1, x1)
    bus_place[0] *= height
    bus_place[2] *= height
    bus_place[1] *= width
    bus_place[3] *= width

    # pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # result = utils.draw_bbox(img, pred_bbox)
    # result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    leftup = (int(bus_place[1]), int(bus_place[0]))
    rightdown = (int(bus_place[3]), int(bus_place[2]))

    return leftup, rightdown
