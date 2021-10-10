import mediapipe as mp
import cv2


class handTrack():

    def __init__(self, mode=False, max_hands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.hands = mp.solutions.hands.Hands(self.mode, self.max_hands,
                                              self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If hand is seen then this will != None
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if (draw):
                    self.mpDraw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                    #creates bounding box for one hand
                    lm = self.findPosition(img)
                    x_min, y_min, x_max, y_max = self.boundingBox(lm, 30)
                    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        return img
    def boundingBox(self, handLM, padding=0):
        x_max, y_max = 0, 0
        x_min, y_min = 1000000000, 1000000000
        for id, x, y in handLM:
            # lm gives x,y, and z location in percentage coordinates
            if (x < x_min):
                x_min = x
            if (x > x_max):
                x_max = x
            if (y < y_min):
                y_min = y
            if (y > y_max):
                y_max = y

        return [x_min-padding, y_min-padding, x_max+padding, y_max+padding]

    def findPosition(self, img, handNum=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                # id gives the point on the hand
                # lm gives x,y, and z location in percentage coordinates
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)  # gives the pixel location
                lmList.append([id, x, y])

        return lmList