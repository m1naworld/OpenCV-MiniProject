import cv2
import numpy as np


# 전처리 함수 구현 : 명암도 영상 변환, 블러링 및 소벨 에지 검출 > 이진화 및 모폴로지 닫힘 연산 수행
def preprocessing(car_no):
    image = cv2.imread("test_car/%02d.jpg" % car_no, cv2.IMREAD_COLOR)
    if image is None:
        print("이미지를 찾을 수 없습니다.")
        return None, None

    # 명암도 영상 변환, 블러링, 수직 에지 검출
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환: BGR -> 단일채널(그레이스케일 = 명암도)
    gray = cv2.blur(gray, (5, 5))  # 5x5 평균 블러링 필터 적용
    # Sobel 에지 검출 -> 1차 미분 연산자 sobel(src, ddepth, dx, dy, ksize)
    # ddeph:결과 이미지 데이터 타입 >> 채널 범위가 달라짐
    # dx, dy : 각각 x방향, y 방향으로 미분 차수 // (1, 0)일때 x방향 미분(수직 마스크),  (0, 1)일때 y방향 미분(수평 마스크)
    # ksize: 확장 Sobel 커널 크기 1, 3, 5, 7 중 하나의 값으로 설정.
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)  # 수직 에지 검출

    # 이진화
    # cv2.threshold(img, threshold, value, type_falg)
    # threshold: 임계값 // value: 임계값 기준에 만족하는 픽셀에 적용할 값 // type_flag: 스레시홀딩 적용 방법
    # THRESH_BINARY: 픽셀 값이 임계값을 넘으면 value로 지정하고, 넘지 못하면 0으로 지정
    th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]  # 어레이만 가져옴
    # 모폴로지 닫힘 연산 (열림연산 : 배경잡음제거에 용이 / 닫힘연산: 객체 내부 잡음 제거에 용이)
    kernel = np.ones((5, 17), np.uint8)  # 번호판 모양과 비슷한 가로로 긴 닫힘 연산 마스크
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel,
                             iterations=3)  # 닫힘 연산 3번 수행

    # 이미지 전처리 확인
    # cv2.imshow("전처리 이미지", morph)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image, morph


# 번호판 후보 영역 판정 함수(번호판 넓이 및 종횡비 검사)
def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0:
        return False

    aspect = h / w if h > w else w / h
    chk1 = 1000 < (h * w) < 30000  # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 8  # 번호판 종횡비 조건
    return chk1 and chk2  # bool값 반환


# 번호판 후보 생성
def find_candidates(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)  # 이진화 이미지에서 윤곽선 검색

    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]  # 회전 사각형 반환
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates


# 후보영상 개선 함수 - 컬러 활용
def color_candidate_img(image, candi_center):
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)  # 채움 행렬
    dif1, dif2 = (25, 25, 25), (25, 25, 25)  # 채움 색상 범위
    flags = 0xff00 + 4 + cv2.FLOODFILL_FIXED_RANGE  # 채움 방향 및 방법
    flags += cv2.FLOODFILL_MASK_ONLY  # 결과 영상만 채움

    # 후보 영역을 유사 컬러로 채우기
    pts = np.random.randint(-15, 15, (20, 2))  # 임의 좌표 20개 생성
    # ^ 코드 실행마다 후보 영역이 바뀐 이유
    pts = pts + candi_center  # 중심좌표로 평행이동
    for x, y in pts:  # 임의 좌표 순회
        if 0 <= x < w and 0 <= y < h:  # 후보 영역 내부 이면
            _, _, fill, _ = cv2.floodFill(image, fill, (x, y), 255, dif1, dif2,
                                          flags)  # 특정영역을 고립시키거나 구분할 때 사용되는 기능
            return cv2.threshold(fill, 120, 255,
                                 cv2.THRESH_BINARY)[1]  # 이진화 이미지로 반환


# 후보영상 각도 보정 함수
def rotate_plate(image, rect):
    center, (w, h), angle = rect  # 중심 좌표, 크기, 회전각도
    if w < h:
        w, h = h, w
        angle = -1  # 이미지들의 번호판 영역이 평균적으로 잘 검출되는 각도로 변경

    size = image.shape[1::-1]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)  # 회전 행렬 계산
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 변환

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)  # 후보 영역 가져오기
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # 명암도 영상
    return cv2.resize(crop_img, (144, 28))  # 크기변경 후 반환


# verify_aspect_size()에 의해 후보영역이 생성되지 않을 경우 대비
while True:  # 번호판 후보 생성 코드

    car_no = int(input("자동차 영상 번호(0~17): "))
    image, morph = preprocessing(car_no)  # 전처리 과정
    candidates = find_candidates(morph)  # 번호판 후보 영역 검색

    fills = [
        color_candidate_img(image, center) for center, _, _ in candidates
    ]  # 후보 영역 재생성
    new_candis = [find_candidates(fill) for fill in fills]  # 재생성 영역 검사
    new_candis = [cand[0] for cand in new_candis if cand]  # 재후보 있으면 저장
    candidate_imgs = [rotate_plate(image, cand)
                      for cand in new_candis]  # 후보 영역 영상 # 리스트

    svm = cv2.ml.SVM_load(
        "SVMTrain.xml")  # 학습된 데이터 적재

    if len(candidate_imgs) != 0:
        rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))  # 1행으로 변환
        _, results = svm.predict(rows.astype("float32"))  # 분류 수행
        global correct
        correct = np.where(results == 1)[0]  # 정답 인덱스 찾기
        break

    elif len(candidate_imgs) == 0:
        print("후보영역을 검출하지 못했습니다. 다시 시도해 주세요.")
        continue

# 후보 영역 확인을 위한 윈도우 화면
# 번호판이 후보영역으로 생성되었지만 번호판으로 판별되지 않았을시 확인용
for i, img in enumerate(candidate_imgs):
    cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True,
                  (0, 255, 255), 2)
    cv2.imshow("candidate_img - " + str(i), img)

# 번호판 검출 성공시 윈도우 화면
for i, idx in enumerate(correct):  # enumerate() 인덱스와 값을 동시에 가져옴
    cv2.imshow("plate image_" + str(i), candidate_imgs[idx])  # 후보영역 영상 출력
    cv2.resizeWindow("plate image_" + str(i), (250, 28))  # 윈도우 크기 조절

for i, candi in enumerate(new_candis):
    color = (0, 255, 0) if i in correct else (
        0, 0, 255)  # 후보영역 색 지정 (번호판 검출 성공시: 초록색, 실패시 : 빨간색)
    cv2.polylines(image, [np.int32(cv2.boxPoints(candi))], True, color,
                  2)  # 후보 영역 표시

print("번호판 검출완료") if len(correct) > 0 else print("번호판 미검출")

cv2.imshow("image", image)
cv2.waitKey(0)

# 이미지에 따른 코드변환
# 4번사진 1/5정도의 확률로 번호판 판별
# 5번사진 def rotate_plate() angle값 -2로 변경시 번호판 판별 확률 ↑
# 7번 angle값 -5로 변경시 확률 ↑
# 9번 닫힘연산 2회 혹은 4회시 확률 ↑
# 10번 13번 잘안됨.
# 14번 닫힘연산 2회시 확률 ↑
# 16번 닫힘연산 4회시 확률 ↑
