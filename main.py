from bus_arrive import get_bus_list
from color_detection import detect_color
from number_detection import detect_number
from door_detection import detect_door
from angle_detection import detect_angle
from calculate_distance_angle import calculate_distance_angle

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import matplotlib.pyplot as plt
import core.utils as utils

MODEL_PATH = './yolov4-416'
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# def main(img_path):
img_path = './input/bus370.png'
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

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=50,
    max_total_size=50,
    iou_threshold=IOU_THRESHOLD,
    score_threshold=SCORE_THRESHOLD
)

door_place = np.zeros(4)

for i in range(len(classes)):
    for j in range(len(classes[i])):
        if classes[i][j] == 5:
#             print(classes[i][j])
#             print()
#             print((boxes[i][j]).numpy())
            door_place += boxes[i][j].numpy()
            
door_place[0] *= height
door_place[2] *= height
door_place[1] *= width
door_place[3] *= width

pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

result = utils.draw_bbox(img, pred_bbox)

result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
plt.imshow(result)
cv2.imwrite('370_bus_yolo.png', result)

height, width, channel = img.shape
temp_result = np.zeros((height, width, channel), dtype=np.uint8)
busx = door_place[3] - door_place[1]
busy = door_place[2] - door_place[0]
centerx = width / 2
centery = height / 2
print(door_place) #[y1, x1, y2, x2]

# for d in possible_contours:
# cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
# cv2.rectangle(temp_result, pt1=int((door_place[0], door_place[1])), pt2=int((door_place[0]+door_place[2])), int(door_place[1])+door_place[3]), color=(255, 255, 255), thickness=2)

cv2.rectangle(temp_result,
            (int(centerx - busx/2), int(centery - busy/2)),
            (int(centerx + busx/2), int(centery + busy/2)),
            (0,0,255), 2)

leftup=(int(centerx - busx/2), int(centery - busy/2))#왼쪽 위 꼭지점 좌표
rightdown=  (int(centerx + busx/2), int(centery + busy/2)) # 오른쪽 위 꼭지점 좌표        
print(leftup, rightdown)  

detected_color = detect_color('370_bus_yolo.png')
#calculate_distance_angle(p1, p4, radian)
#detect_number(number, img_ori):
#color_detection(image_path):
#detect_door(img, leftup, rightdown )
# detect_angle(img, leftup, rightdown)

#TODO : 반복문으로; get_bus_list 의 반환값을 가정
busList = { 
    #3번이 파란색
    '370': '3',
    '7212': '4',
    '642': '3'
}

num_bus =0

for num,color in busList.items():
    if color == detected_color:
        num_bus+=1

if num_bus == 0:
    print('No process required')

if num_bus >= 2 and detect_number(img_path,'370', (111, 56), (1058, 721)) == False:
    print('Number detection fail')

door = detect_door('370_bus_yolo.png', leftup, rightdown)
radian = detect_angle(img_path, door)
p1= np.array((door['x'], door['y']))
p4= np.array((door['x']+door['w'], door['y']+door['h']))
distance, angle = calculate_distance_angle(p1, p4, radian)
print(distance, angle)