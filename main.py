import cv2
from hand_track import handTrack

cam = cv2.VideoCapture(0)

hands = handTrack()

while True:
    success, img = cam.read()
    hands.findHands(img)
    #print(hands.findPosition(img))

    cv2.imshow("Camera", img)
    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()