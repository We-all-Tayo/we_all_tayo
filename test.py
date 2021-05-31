from bus_arrive import BusArrive
from color_detection import ColorDetection
from number_detection import NumberDetection
from door_detection import DoorDetection
from angle_detection import AngleDetection
from calculate_distance_angle import Calculator

import numpy as np

bus_arrive = BusArrive()
color_detection = ColorDetection()
number_detection = NumberDetection()
door_detection = DoorDetection()
angle_detection = AngleDetection()
calculator = Calculator()
################################################################################
#XXX Assume the input data as...

target_bus = '4211'
target_station = '23322'
input_image = './input/bus4211.jpg'
leftup = (963, 127)
rightdown = (3383, 1799)
################################################################################
#XXX TEST CODES -- bus_arrive.get_bus_list

if target_bus in bus_arrive.get_bus_dict(target_station):
    print(target_bus, 'is comming...')
else:
    print(target_bus, 'is NOT comming...')
################################################################################
#XXX TEST CODES -- color_detection.detect_color

# Red : 6 광역 | Green: 4 지선 | Blue: 3 간선 | Yellow: 5 순환
print('bus type:', color_detection.detect_color(input_image, leftup=leftup, rightdown=rightdown))
################################################################################
#XXX TEST CODES -- number_detection.detect_number

if number_detection.detect_number(input_image, target_bus, leftup, rightdown):
    print('number detection success')
else:
    print('number detection fail')
################################################################################
#XXX TEST CODES -- door_detection.detect_door

door = door_detection.detect_door(input_image, leftup, rightdown)
#print('door output:', door)
################################################################################
#XXX TEST CODES -- angle_detection.detect_angle

radian = angle_detection.detect_angle(input_image, leftup, rightdown)
print('detected angle:', radian)
################################################################################
#XXX TEST CODES -- calculate_distance_angle.calculate_distance_angle

distance, angle = calculator.calculate_distance_angle(door, radian, input_image)
print('calculate:', distance, angle)
################################################################################
#XXX RESULT VALUES

print(
    str(round(distance / 1000, 2)) + " meter",
    str(round(angle * 180 / np.pi)) + " 도",
)