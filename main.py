import cv2
from hand_track import handTrack
from sign_language_nn import SLNN

cam = cv2.VideoCapture(0)

hands = handTrack()
#model = SLNN()

LEFT_HAND = 0
RIGHT_HAND = 1

preference_hand = RIGHT_HAND

while True:
    success, img = cam.read()
    hands.findHands(img, draw=False)
    num_hands = hands.num_visible_hands
    handLM = []

    if (num_hands == 1):
        #Whichever hand is in the image
        handLM = hands.findPosition(img)
    elif (num_hands == 2):
        #Hand Preference
        handLM = hands.findPosition(img, handNum=preference_hand)

    if (not handLM):
        # Checking for 0 hands doesn't help once it sees one hand
        # for the first time.
        # Doesn't move forward and repeats reading from camera
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            break
        continue

    #Dims can be negative based on the NN so if it crashes that's why
    hand_img_dims = hands.boundingBox(handLM)

    #Projects img of hand (y start -> y end, x start -> x end)
    hand_img = img[hand_img_dims[1]:hand_img_dims[3], hand_img_dims[0]:hand_img_dims[2]]
    hand_img = cv2.resize(hand_img, (28,28))
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Camera", hand_img)
    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()