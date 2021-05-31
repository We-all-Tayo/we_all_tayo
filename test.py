from bus_arrive import get_bus_dict
from color_detection import detect_color
from number_detection import detect_number
from door_detection import detect_door
from angle_detection import detect_angle
from calculate_distance_angle import calculate_distance_angle

import numpy as np

################################################################################
#XXX Assume the input data as...

target_bus = '4211'
target_station = '23322'
input_image = './input/bus4211.jpg'
leftup = (963, 127)
rightdown = (3383, 1799)
################################################################################
#XXX TEST CODES -- bus_arrive.get_bus_list

if target_bus in get_bus_dict(target_station):
    print(target_bus, 'is comming...')
else:
    print(target_bus, 'is NOT comming...')
################################################################################
#XXX TEST CODES -- color_detection.detect_color

# Red : 6 광역 | Green: 4 지선 | Blue: 3 간선 | Yellow: 5 순환
print('bus type:', detect_color(input_image, leftup=leftup, rightdown=rightdown))
################################################################################
#XXX TEST CODES -- number_detection.detect_number

if detect_number(input_image, target_bus, leftup, rightdown):
    print('number detection success')
else:
    print('number detection fail')
################################################################################
#XXX TEST CODES -- door_detection.detect_door

door = detect_door(input_image, leftup, rightdown)
#print('door output:', door)
################################################################################
#XXX TEST CODES -- angle_detection.detect_angle

radian = detect_angle(input_image, leftup, rightdown)
print('detected angle:', radian)
################################################################################
#XXX TEST CODES -- calculate_distance_angle.calculate_distance_angle

distance, angle = calculate_distance_angle(door, radian, input_image)
print('calculate:', distance, angle)
################################################################################
#XXX RESULT VALUES

print(
    str(round(distance / 1000, 2)) + " meter",
    str(round(angle * 180 / np.pi)) + " 도",
)