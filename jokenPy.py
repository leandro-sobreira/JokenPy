import mediapipe as mp
import numpy as np
import cv2
import cvzone
import math
from datetime import datetime

video = cv2.VideoCapture(1)
hand = mp.solutions.hands

Hand = hand.Hands(max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.1)
mpDraw = mp.solutions.drawing_utils

spStop = background = cv2.imread("sprite/sprite0.png", cv2.IMREAD_UNCHANGED)
spPaper = background = cv2.imread("sprite/sprite1.png", cv2.IMREAD_UNCHANGED)
spRock = background = cv2.imread("sprite/sprite2.png", cv2.IMREAD_UNCHANGED)
spScizor = background = cv2.imread("sprite/sprite3.png", cv2.IMREAD_UNCHANGED)

def distEu(v1, v2):
	dim, soma = len(v1), 0
	for i in range(dim):
		soma += math.pow(v1[i] - v2[i], 2)
	return math.sqrt(soma)

timer = 0

while True:
    check, img = video.read()
    #imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.flip(img,1)
    results = Hand.process(img)
    handsPoints = results.multi_hand_landmarks
    h, w,_ = img.shape
    pontos = []

    if handsPoints:
        for points in handsPoints:
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)

            for id,cord in enumerate(points.landmark):
                cx, cy = int(cord.x*w), int(cord.y*h)
                cv2.putText(img, str(id), (cx,cy+10), cv2.FONT_HERSHEY_PLAIN, 0.35, (255, 0, 0), 1)
                pontos.append((cx,cy))
        
        dedos = [8, 12, 16, 20]
        Fingers = [False, False, False, False, False]

        if points:
            if (distEu(pontos[4], pontos[17]) > distEu(pontos[2], pontos[17])):
                  Fingers[0] = True
            finger = 0
            for x in dedos:
                finger += 1
                if (distEu(pontos[x], pontos[0]) > distEu(pontos[x-2], pontos[0])):
                  Fingers[finger] = True

            #UI para status do dados
            for x in range(5):
                if Fingers[x]:
                    cv2.rectangle(img, (10+(10*x),10), (20+(10*x),20), (0,255,0), cv2.FILLED)
                else:
                    cv2.rectangle(img, (10+(10*x),10), (20+(10*x),20), (100,100,100), cv2.FILLED)
            
            #if not (pontos[8][0] > 40 and pontos[8][0] < 160 and pontos[8][1] > 40 and pontos[8][1] < 80  or timer > 0):
            #    cv2.rectangle(img, (40,40), (160,80), (0,255,0), cv2.FILLED)
            #    cv2.putText(img, "Start", (80,70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            #    startTimer = datetime.now()

            #else:
            #    timer = 10-(datetime.now()-startTimer).seconds
            #    if timer <= 10 and timer > 5:
            #        cv2.putText(img, str(timer-5), (80,70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            #    else:
            if not Fingers[1] and Fingers[2] and not Fingers[3] and not Fingers[4]:
                cv2.putText(img, "._.", (80,70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            elif Fingers[0] and Fingers[1] and Fingers[2] and Fingers[3] and Fingers[4]:
                img = cvzone.overlayPNG(img, spPaper, [80,70])
            elif not Fingers[0] and not Fingers[1] and not Fingers[2] and not Fingers[3] and not Fingers[4]:
                img = cvzone.overlayPNG(img, spRock, [80,70])
            elif Fingers[1] and Fingers[2] and not Fingers[3] and not Fingers[4]:
                img = cvzone.overlayPNG(img, spScizor, [80,70])
            else:
                img = cvzone.overlayPNG(img, spStop, [80,70])
                    
    cv2.imshow("Imagem", img)
    cv2.waitKey(1)


