I Wanna Take a Bus
===

3개월 전 시각 장애인이 “시각 장애인 혼자 버스를 탈 수 있을까?” 라는 제목으로 유튜브에 동영상을 게시 하였습니다. 무려 168만 회의 조회 수를 기록하고 많은 사람들에게 시각 장애인들의 버스 이용 어려움을 깨달을 수 있게 한 영상이었습니다. 많은 버스정류장에서 어떤 버스가 도착하는지 음성 안내를 해주고 있지만, 시각장애인들은 단순히 이 소리만 듣고는 버스를 이용할 수가 없다는 것을 알았습니다. 버스정류장 앞에는 버스만 지나가는 것이 아닐 뿐더러 버스가 정차를 하더라도 버스의 탑승문이 어디 있는지 알 수가 없기 때문에 누구의 도움 없이는 버스를 탑승할 수 없었습니다. 따라서 저희는 비전 인식으로 ‘어떤 버스’ 가 정차를 하는지 인식을 하고 ‘버스의 출입문’ 까지의 거리를 계산하는 인공지능 모델을 구상하였습니다. 

프로젝트 구조
---

```
.
├── angle_detection.py # 허프 변환을 통해 버스 경계선을 찾고 그 각도를 계산한다
├── bus_arrive.py # 버스 정보 API를 통해 현재 도착 버스 노선번호와 타입 목록을 불러온다
├── calculate_distance_angle.py # 주어진 좌표와 각도를 통해 버스까지의 거리와 각도를 유추한다
├── checkpoints/ # YOLOv4 라이브러리
├── color_detection.py # 주어진 이미지 내 가장 큰 비율을 차지하는 색상을 통해 버스 타입을 유추한다
├── core/ # Tensorflow YOLOv4 core 라이브러리
├── data/ # YOLOv4 훈련 데이터
├── door_detection.py # 주어진 버스 이미지에서 버스 출입문 위치를 유추한다
├── number_detection.py # 주어진 버스 이미지의 노선 번호를 유추한다
├── main.py 
└── test.py # 각 모듈에 대한 유닛 테스트

```

의존성
---

```bash
$ pip3 install easydict imutils matplotlib numpy==1.19.2 opencv-python pytesseract python-dotenv scipy tensorflow==2.4.1 urllib3
```

- [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

```bash
$ sudo apt install tesseract-ocr-kor
$ export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata/"
$ # 이는 Ubuntu 20.04 기준이며, 운영체제에 따라 차이가 있을 수 있습니다.
```
