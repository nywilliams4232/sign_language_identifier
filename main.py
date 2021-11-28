import cv2
import numpy as np
from hand_track import handTrack
from data_loader import loadData
from sign_language_nn import SLCNN
from tensorflow.math import confusion_matrix

characters = ["A", "B", "C", "D", "E", "F", "G",
                  "H", "I","J", "K", "L", "M", "N", "O",
                  "P", "Q", "R", "S", "T", "U", "V",
                  "W", "X", "Y", "Z"]
#def create_data_set():


def train_model():
    # example of data loading
    y_train, X_train = loadData("sign_mnist_train.csv")

    print(X_train[0])

    model = SLCNN()
    for i in range(10):
        guass = np.random.normal(0, 0.15, (28, 28, 1))
        x_train_noise = X_train + guass
        x_train_invert = 1 - x_train_noise
        x_train_final = np.append(x_train_noise, x_train_invert).reshape(-1, 28,28,1)
        y_train_final = np.append(y_train, y_train)
        model.fit(x_train_final, y_train_final, epochs=1, batch_size=100)
    #model.fit(X_test, y_test, epochs=2, batch_size=100)

    model.save_weights("weights_slnn4.w")

    return model

def test_model():
    y_test, X_test = loadData("sign_mnist_test.csv")

    model = SLCNN()
    model.load_weights()

    for i in range(100):
        im = cv2.resize(X_test[i], (600, 600))
        cv2.imshow("Image", im)

        print("Answer:", characters[y_test[i]])

        print("Prediction:", characters[np.argmax(model.predict(X_test)[i])])
        cv2.waitKey()

def main():
    #train_model()
    #test_model()
    y_train, X_train = loadData("sign_mnist_train.csv")
    y_test, X_test = loadData("sign_mnist_test.csv")
    print(y_train.shape)
    cam = cv2.VideoCapture(0)
    model = SLCNN()
    model.load_weights("weights_slnn4.w")
    hands = handTrack()

    #print(model.model.summary())
    #model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    #model.save_weights()
    #y_predict = model.predict(X_test)


    #print(confusion_matrix(y_test, np.argmax(y_predict, axis=1), num_classes=26, weights=None,name=None))

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

        if(handLM != []):
            #Dims can be negative based on the NN so if it crashes that's why
            hand_img_dims = hands.boundingBox(handLM, padding=20)
            #Projects img of hand (y start -> y end, x start -> x end)
            hand_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hand_img = hand_img[hand_img_dims[1]:hand_img_dims[3], hand_img_dims[0]:hand_img_dims[2]]
            hand_img = cv2.resize(hand_img, (28,28))

            hand_img = np.array(hand_img)/255.0
            cv2.imshow("Image", cv2.resize(hand_img, (600, 600)))
            hand_img = np.reshape(hand_img, (-1, 28, 28, 1))

            predict_arr = model.predict(hand_img)[0]
            predicted_char = characters[np.argmax(predict_arr)]
            print("Prediction:", predicted_char)
            img = cv2.rectangle(img, (hand_img_dims[0], hand_img_dims[1]), (hand_img_dims[2], hand_img_dims[3]), color = (255,0,0))
            print(predict_arr)
            dtype = [('Letter', 'S1'), ('prob', float)]
            dict = {}
            for A, B in zip(characters, predict_arr):
                dict[A] = B

            data = list(dict.items())
            a = np.array(data, dtype=dtype)
            a = np.sort(a, order='prob')
            for i, val in enumerate(np.flip(a[-5:])):
                img = cv2.putText(img, str(val[0]) + f' {val[1]:05f}', (hand_img_dims[2], hand_img_dims[3] - 100 + i*20), color=(0,255,0), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1)


        cv2.imshow("Camera", img)

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()


#train_model()
main()
#test_model()