import math
import cv2
import numpy as np
import statistics

class AngleDetection:
    def detect_angle(self, image_path, bus_leftup, bus_rightdown):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)

        lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 20,
                                minLineLength=(bus_rightdown[0] - bus_leftup[0]) / 4, maxLineGap=200)
        lines2 = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 60,
                                minLineLength=(bus_rightdown[0] - bus_leftup[0]) / 4, maxLineGap=200)
        lines3 = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 120,
                                minLineLength=(bus_rightdown[0] - bus_leftup[0]) / 4, maxLineGap=200)
        lines = np.append(lines, lines2, axis=0)
        lines = np.append(lines, lines3, axis=0)

        angles = []
        for line in lines:
            # 왼쪽 아래 꼭지점 (line[0][0], line[0][1]), 오른쪽 위 꼭지점 (line[0][2], line[0][3])
            # 왼쪽 아래 꼭지점과 오른쪽 위 꼭지점의 y값들이 버스의 상반신에 있어야한다
            if 0 < line[0][1] < (bus_leftup[1] + bus_rightdown[1]) / 2 and 0 < line[0][3] < \
                    (bus_leftup[1] + bus_rightdown[1]) / 3 \
                    and 0 < line[0][0] < bus_rightdown[1] * 0.8 and line[0][3] > bus_leftup[1]:
                # 왼쪽 아래 꼭지점과 오른쪽 위 꼭지점의 x값들이 버스의 내부에 있어야한다
                if line[0][0] > bus_leftup[0] * 0.8 and line[0][2] < bus_rightdown[0]:
                    # 직선의 기울기가 0도에 가까울경우 스킵
                    if line[0][3] == line[0][1] or abs(line[0][1] - line[0][3]) < 20:
                        continue
                    angle = math.atan((line[0][2] - line[0][0]) / (line[0][1] - line[0][3]))
                    # 버스 측면 기울기의 각도가 y축기준 1 radian(57.2도) 에서 1.4 radian(80.2도) 사이에 있어야한다
                    if 1 < angle < 1.4:
                        if line[0][0] < line[0][2] < bus_rightdown[0] * 0.8 and line[0][1] > line[0][3]:
                            angles.append(round(angle, 2))

        return statistics.mode(angles)
