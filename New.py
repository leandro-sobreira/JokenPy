import cv2
import cvzone
import mediapipe as mp
import math

def distEu(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

spStop = cv2.imread("sprite/sprite0.png", cv2.IMREAD_UNCHANGED)
spPaper = cv2.imread("sprite/sprite1.png", cv2.IMREAD_UNCHANGED)
spRock = cv2.imread("sprite/sprite2.png", cv2.IMREAD_UNCHANGED)
spScizor = cv2.imread("sprite/sprite3.png", cv2.IMREAD_UNCHANGED)

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.1) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    h, w, _ = image.shape
    pontos = []

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for points in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            points,
            mp_hands.HAND_CONNECTIONS)
        for id, cord in enumerate(points.landmark):
            cx, cy = int(cord.x * w), int(cord.y * h)
            cv2.putText(image, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_PLAIN, 0.35, (255, 0, 0), 1)
            pontos.append((cx, cy))

      dedos = [8, 12, 16, 20]
      Fingers = [False, False, False, False, False]

      if points:
          if distEu(pontos[4], pontos[17]) > distEu(pontos[2], pontos[17]):
              Fingers[0] = True
          finger = 0
          for x in dedos:
              finger += 1
              if distEu(pontos[x], pontos[0]) > distEu(pontos[x - 2], pontos[0]):
                  Fingers[finger] = True

          # UI para status do dados
          for x in range(5):
              if Fingers[x]:
                  cv2.rectangle(image, (10 + (10 * x), 10), (20 + (10 * x), 20), (0, 255, 0), cv2.FILLED)
              else:
                  cv2.rectangle(image, (10 + (10 * x), 10), (20 + (10 * x), 20), (100, 100, 100), cv2.FILLED)

          # if not (pontos[8][0] > 40 and pontos[8][0] < 160 and pontos[8][1] > 40 and pontos[8][1] < 80  or timer > 0):
          #    cv2.rectangle(image, (40,40), (160,80), (0,255,0), cv2.FILLED)
          #    cv2.putText(image, "Start", (80,70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
          #    startTimer = datetime.now()

          # else:
          #    timer = 10-(datetime.now()-startTimer).seconds
          #    if timer <= 10 and timer > 5:
          #        cv2.putText(image, str(timer-5), (80,70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
          #    else:
          if not Fingers[1] and Fingers[2] and not Fingers[3] and not Fingers[4]:
              cv2.putText(image, "._.", (80, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
          elif Fingers[0] and Fingers[1] and Fingers[2] and Fingers[3] and Fingers[4]:
              image = cvzone.overlayPNG(image, spPaper, [80, 70])
          elif not Fingers[0] and not Fingers[1] and not Fingers[2] and not Fingers[3] and not Fingers[4]:
              image = cvzone.overlayPNG(image, spRock, [80, 70])
          elif Fingers[1] and Fingers[2] and not Fingers[3] and not Fingers[4]:
              image = cvzone.overlayPNG(image, spScizor, [80, 70])
          else:
              image = cvzone.overlayPNG(image, spStop, [80, 70])


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()