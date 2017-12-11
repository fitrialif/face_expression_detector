import cv2
import model
import os
import numpy as np

face_expression_detector = model.Model()
def get_emotion_by_index(index):
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    if index ==0:
        return "Angry"
    elif index ==1:
        return "Disgust"
    elif index ==2:
        return "Fear"
    elif index ==3:
        return "Happy"
    elif index ==4:
        return "Sad"
    elif index ==5:
        return "Surprise"
    elif index ==6:
        return "Neutral"
    else:
        return "Unregistered emotion"
while True:
    pic = cv2.imread(os.path.join("data_set","test6.jpg"))
    grayed = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    grayed_height, grayed_width = grayed.shape[:2]
    desired_h, desired_w = 48, 48
    resized_ratio_h, resized_ratio_w = desired_h / grayed_height, desired_w / grayed_width
    res = cv2.resize(grayed, None, fx=resized_ratio_w, fy=resized_ratio_h, interpolation=cv2.INTER_CUBIC)
    res = np.reshape(res, (-1, 2304))
    print(res.shape)
    checkpoint_save_dir = os.path.join("checkpoint")

    feed_dict = {face_expression_detector.X: res, face_expression_detector.keep_prob: 1}
    print(get_emotion_by_index(face_expression_detector.predict_result(checkpoint_save_dir, feed_dict)))
    cv2.imshow("",pic)
    cv2.waitKey(1)