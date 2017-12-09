import numpy as np
import cv2
import model
import os

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
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap= cv2.VideoCapture(0)
face_expression_detector = model.Model()
while True:
    ret,frame = cap.read()
    grayed = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayed,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = grayed[y:y+h,x:x+w]
        roi_color = grayed[y:y+h,x:x+w]
        eyes= eye_cascade.detectMultiScale(roi_gray)
        for e_x,e_y,e_w,e_h in eyes:
            cv2.rectangle(roi_color,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)

            grayed_height, grayed_width = grayed.shape[:2]
            desired_h, desired_w = 48, 48
            resized_ratio_h, resized_ratio_w = desired_h / grayed_height, desired_w / grayed_width
            res = cv2.resize(grayed, None, fx=resized_ratio_w, fy=resized_ratio_h, interpolation=cv2.INTER_CUBIC)
            res = np.reshape(res,(-1,2304))
            print(res.shape)
            checkpoint_save_dir = os.path.join("checkpoint")

            feed_dict = {face_expression_detector.X:res,face_expression_detector.keep_prob:1}
            print(get_emotion_by_index(face_expression_detector.predict_result(checkpoint_save_dir,feed_dict)))
    cv2.imshow("frame",grayed)
    if(cv2.waitKey(1) &0xff == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()