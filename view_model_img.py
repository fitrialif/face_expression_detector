import cv2
import os
import numpy as np
from model import *;


file_path  = os.path.join("data_set","fer2013.csv")
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
a = Model()
checkpoint_save_dir = a.checkpoint_save_dir

for x,y,purpose in a.get_input(file_path,10,5):
    feed_dict = {a.X: (x), a.keep_prob: 1}
    for x_each,y_each,purpose_each in zip(x,y,purpose):
        while True:
            x_each= np.array(x_each).reshape(48,48)
            frame = cv2.resize(x_each, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            frame = cv2.putText(frame,get_emotion_by_index(int(y_each)),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
            cv2.imshow("",frame)

            if(cv2.waitKey(1) &0xff == ord('q')):
                cv2.destroyAllWindows()
                break