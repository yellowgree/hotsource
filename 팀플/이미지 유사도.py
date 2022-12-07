'''
    File Comment
    
    aythor : 202234898 박하은
    date : 11/30
    
    summary : 유신이 코멘트 보고 수정 사항 적용
    
            내가 제작한 함수 : image_similarity()
            
            두 가지 이미지를 가지고 같은 사진인지 다른 사진인지 구별해주는 함수
            5가지의 기준을 가지고 엄밀하게 구별하기 때문에 뛰어난 유사도 분석이 가능함

            만약에 두 이미지가 일치한다면 0을, 일치하지 않는다면 1을 리턴함.
            

    idea :  주제가 틀린 그림 찾기이고 각각 함수를 구현해서 합치는 방향으로 가기로 했는데 구체적으로
            필요한 기능을 생각해봄. 내가 유사도 분석으로 사진이 같은지 다른지 판별해주는 함수를 구현했으니까
            서로 똑같은 사진과 서로 약간의 차이가 있는 사진 모두 필요함.

            그러니까 하나의 사진을 가지고 거기서 일부분을 변형하는 함수가 따로 있으면 좋을 것 같음.
            수업시간에 배운 내용 응용하면 쉽게 구현할 수 있을 것 같음. ex: image_deformation()

            그래서 몇장의 사진들을 임의로 저장해서 리스트에 넣으면 거기서 랜덤으로 몇개를 image_deformation 함수로
            변형 시킨 다음에 내가 만든 image_similarity 함수로 먼저 유사도로 판단하고, 만약에 서로 다른 부분이
            있는 사진이라면 거기서 정확히 어디 부분이 다른지 체크해주는 함수를 만드는 방향. ex) image_pointCheck()


            그러면 틀린 그림이 없는 사진일수도 있는거지... 더 어려워질 수 있음


            그 밖에도 시간 제한이나.. 목숨이나.. 점수나..그런거 결정하는 기능들도 필요할듯.


            이미지 변형이 되든, 변형이 안되든 사진은 두 쌍씩 묶어서 저장하는 게 나중에 출력할 때 편할 것 같은데
            딕셔너리 기능 활용하면 좋을듯. 딕셔너리가 key:value 이긴 한데 어차피 이미지는 문자열로 저장할거고 여기까지 생각해봄
            

'''
import cv2
import numpy as np
import matplotlib.pylab as plt


def image_similarity():

    imgs = []
    imgs.append(cv2.imread('test1-1.jpg'))
    imgs.append(cv2.imread('test1-2.jpg'))

    hists = []

    for img in imgs:
        #BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        # 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        # hists 리스트에 저장
        hists.append(hist)

    method = 'CHISOR'
    query = hists[0]
    ret = cv2.compareHist(query, hists[1], method)

    if method == cv2.HISTCMP_INTERSECT:
        ret = ret/np.sum(query)   

    if ret == 1:
        return 0
    else:
        return 1
    

        
image_similarity()



