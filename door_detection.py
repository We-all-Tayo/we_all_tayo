import cv2

MIN_RATIO, MAX_RATIO = 0.01, 0.5

class DoorDetection:
    def detect_door(self, img, leftup, rightdown):
        img_ori = cv2.imread(img)
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

        height, width, _ = img_ori.shape
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9
        )
        contours, _ = cv2.findContours(
            img_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        min_area = height * width * 0.00025
        min_width, min_height = height * width * 0.00001, height * width * 0.00005
        right_x = (rightdown[0] - leftup[0]) / 2
        max_x = rightdown[0] * 0.8  # 문이 바운더리 밖에 있지는 않으니까

        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']

            if area > min_area \
                    and d['w'] > min_width and d['h'] > min_height \
                    and MIN_RATIO < ratio < MAX_RATIO \
                    and max_x > d['x'] > right_x:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)

        possible_contours.sort(key=lambda x: x['h'], reverse=True)
        prob_door = possible_contours[0]

        cv2.rectangle(img_ori, pt1=(prob_door['x'], prob_door['y']),
                    pt2=(prob_door['x'] + prob_door['w'], prob_door['y'] + prob_door['h']),
                    color=(255, 255, 255), thickness=2)

        return prob_door
