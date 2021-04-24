import cv2
import numpy as np
import pytesseract

MIN_AREA = 100
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 0.9

PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 0 #3
MAX_PLATE_RATIO = 10 #10

AREA_MIN_RATIO = 1.3 # 숫자의 비율(숫자 옆에 노이즈 까지 고려해서)
AREA=300 #크기(작은거 없애려고)

MAX_DIAG = 10

MAX_DIAG_MULTIPLYER = 3# 5
MAX_ANGLE_DIFF = 12.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 2 # 3

def preprocessing(image):
    height, width, channel = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
    #값 조정
        blockSize=19, 
        C=-9
    )
    img_blurred = cv2.bilateralFilter(img_thresh, -1, 10, 5)

    # Find contours
    contours, _ = cv2.findContours(
        img_blurred, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(
        temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    
    # Contours Box
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            temp_result, pt1=(x, y), pt2=(x+w, y+h), 
            color=(255, 255, 255), thickness=1)
        
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    return img_thresh, contours_dict

def get_candidates(image, contours_dict, right_x, up_y, max_x):
    height, width, channel = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Candidates
    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO \
                and d["y"] < up_y \
                and max_x > d["x"] > right_x:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    
    # visualize possible contours
    #temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(
            gray, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), 
            color=(255, 255, 255), thickness=2)
    
    # Find the contours with the characters
    result_idx = find_chars(possible_contours, possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for r in matched_result:
        for d in r:
    #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(
                temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), ###DEBUG
                color=(255, 255, 255), thickness=2)
    return matched_result

def calculate_plate_index(contour, plate_min_x, plate_min_y, plate_max_x, plate_max_y):
    x, y, w, h = cv2.boundingRect(contour)
    
    area = w * h
    ratio = w / h

    if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        if x < plate_min_x:
            plate_min_x = x
        if y < plate_min_y:
            plate_min_y = y
        if x + w > plate_max_x:
            plate_max_x = x + w
        if y + h > plate_max_y:
            plate_max_y = y + h

    return plate_min_x, plate_min_y, plate_max_x, plate_max_y

def parse_candidate(image, matched_result, img_thresh):
    height, width, channel = image.shape
    plate_imgs = []
    plate_infos = []
    for _, matched_chars in enumerate(matched_result):
        plate_images(matched_chars, plate_imgs, plate_infos, img_thresh, width, height)

    img_results = []

    for _, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            plate_min_x, plate_min_y, plate_max_x, plate_max_y = calculate_plate_index(contour, plate_min_x, plate_min_y, plate_max_x, plate_max_y)
                    
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        img_results.append(img_result)
    
    return plate_imgs, img_results

def find_chars_contour(d1, d2, matched_contours_idx):
    dx = abs(d1['cx'] - d2['cx'])
    dy = abs(d1['cy'] - d2['cy'])

    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

    distance = np.linalg.norm(
        np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
    
    if dx == 0:
        angle_diff = 90
    else:
        angle_diff = np.degrees(np.arctan(dy / dx))
        
    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
    width_diff = abs(d1['w'] - d2['w']) / d1['w']
    height_diff = abs(d1['h'] - d2['h']) / d1['h']

    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
        matched_contours_idx.append(d2['idx'])

def find_chars(possible_contours, contour_list):
    matched_result_idx = []
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            find_chars_contour(d1, d2, matched_contours_idx)

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(possible_contours, unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx

def plate_images(matched_chars, plate_imgs, plate_infos, img_thresh, width, height):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    
    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO \
            or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        return
    
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })

def filter_numbers(len_number, plate_imgs, channel):
    temp_plate = [] 
    possible_plate = []# 너무 작은 박스랑 박스가 4개 이하인 것들을 걸렀음.

    for i in range(len(plate_imgs)):
        rec_num = 0
        height, width = plate_imgs[i].shape
        
        contours, _ = cv2.findContours(plate_imgs[i], mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) 
        temp_result = np.zeros((height,width, channel), dtype=np.uint8)
        
        temp = cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))    
        temp_result2 = np.zeros((height,width, channel), dtype=np.uint8)
                                
        rect =[]                     
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
        
            ratio = h / w
            area = w * h
            if ratio < AREA_MIN_RATIO or area < AREA:
                continue
            
            bound = (x, y, w, h)
            
            rect.append(bound)
            rec_num += 1
            cv2.rectangle(
                temp_result2, pt1=(x, y), pt2=(x+w, y+h), 
                color=(255, 0, 0), thickness=1)
            
        if rec_num >= len_number:
            temp_plate.append(rect)
            possible_plate.append(plate_imgs[i])    

    return temp_plate, possible_plate

def clustering(index, temp_rec, i, img_index, len_number, x_sorted_plate):
    
    if(index == len(x_sorted_plate[i])-1):
        diag = - (x_sorted_plate[i][index-1][0]
            + x_sorted_plate[i][index-1][2] 
            - x_sorted_plate[i][index][0])
        if diag <= MAX_DIAG:
            temp_rec.append(x_sorted_plate[i][index])
        if len(temp_rec) >= len_number:
            new_img = {'index':i, 'rec':temp_rec}
            img_index.append(new_img)
        return
    
    diag = x_sorted_plate[i][index+1][0] - x_sorted_plate[i][index][0] - x_sorted_plate[i][index][2]

    if diag > MAX_DIAG:
        if len(temp_rec) >= len_number:
            new_img = {'index':i, 'rec':temp_rec}
            img_index.append(new_img)
            
        new_temp = []
        clustering(index + 1, new_temp, i, img_index, len_number, x_sorted_plate)
        
    else:
        temp_rec.append(x_sorted_plate[i][index])
        clustering(index + 1, temp_rec, i, img_index, len_number, x_sorted_plate)

def ocr(plate_img, img_results, plate_chars, i, longest_idx, longest_text):
    chars = pytesseract.image_to_string(plate_img, lang='kor', config='--psm 10')
    img_results.append(plate_img)

    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print('result chars: ', result_chars)
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

def detect_number(image, number, leftup, rightdown):
    image = cv2.imread(image)
    height, width, channel = image.shape
    
    img_thresh, contours_dict = preprocessing(image)
    
    right_x = (871-88)*0.1
    up_y = (527-112)
    max_x = 871*0.8 #문이 바운더리 밖에 있지는 않으니까
    # (88, 112) (871, 527)

    matched_result = get_candidates(image, contours_dict, right_x, up_y, max_x)
    plate_imgs, img_results = parse_candidate(image, matched_result, img_thresh)
        
    len_number = len(number)
    temp_plate, possible_plate = filter_numbers(len_number, plate_imgs, channel)

    # Clustering
    x_sorted_plate = []
    img_index = []
    for i in range(len(temp_plate)):
        x_sorted_plate.append(sorted(temp_plate[i],key=lambda rect: rect[0]))

    for i in range(len(x_sorted_plate)):
        index =0
        temp_rec=[]        
        clustering(index, temp_rec, i, img_index, len_number, x_sorted_plate)

    # Crop the contour of the numbers
    clopped_imgs=[]
    for i in range(len(img_index)):
        first_rec = img_index[i]['rec'][0]
        last_rec = img_index[i]['rec'][-1]
        width = last_rec[0] + last_rec[2] - first_rec[0]
        height,_ =  possible_plate[img_index[i]['index']].shape
        center_x = int(width / 2 + first_rec[0]) 
        center_y = int(height / 2)
        img_cropped = cv2.getRectSubPix(
            possible_plate[img_index[i]['index']], 
            patchSize=(width, height),
            center=(center_x,center_y))

        clopped_imgs.append(img_cropped)
    
    k = 0
    plate_chars = []
    longest_idx, longest_text = -1, 0
    for plate_img in clopped_imgs:
        ocr(plate_img, img_results, plate_chars, i, longest_idx, longest_text)

        k += 1

    for i in plate_chars:    
        temp = i.find(number)
        if temp != -1:
            print(i[temp:len_number+1])
            return True
        else:
            return False
