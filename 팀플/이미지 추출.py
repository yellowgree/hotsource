#유사도 파악하고 유사도가 임계치보다 작을 시
#틀린 부분 출력하기
import cv2
from PIL import ImageChops
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

img1 = cv2.imread('text1-1.jpg')
img2 = cv2.imread('text1-2.jpg')
imgs = [img1, img2]

hists = []

for img in imgs:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    hists.append(hist)

query = hists[0]

method = cv2.HISTCMP_BHATTACHARYYA

for i, histogram in enumerate(hists):
    ret = cv2.compareHist(query, histogram, method)

    if method == cv2.HISTCMP_INTERSECT:
        ret = ret / np.sum(query)

    print("img%d :%7.2f" % (i + 1, ret), end='\t')

if ret <= 0.95:
    print('Find difference')

    scr = Image.open('../../../../pythonProject1/text1-1.jpg')
    dest = Image.open('../../../../pythonProject1/text1-2.jpg')

    diff = ImageChops.difference(scr, dest)
    diff.save('diff.jpg')

    diff_img = cv2.imread('diff.jpg')

    cv2.imshow('diff', diff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif ret >= 0.95 and ret <= 1:
    print('Same image')
else:
    print('Image Error')
